import ccxt
import pandas as pd
import pandas_ta as ta
from backtesting import Backtest, Strategy

# ======== CONFIGURATION (modifiable) =========
EXCHANGE_ID       = 'delta'            # CCXT id ('delta', 'binance', etc.)
API_KEY           = 'YOUR_API_KEY'     # optional
SYMBOL            = 'XRP/USDT'
TIMEFRAME         = '1d'
BACK_DAYS         = 30
COMMISSION_MAKER  = 0.0002             # 0.02%
COMMISSION_TAKER  = 0.0005             # 0.05%
LOT_SIZE          = 1
STOP_LOSS_PCT     = 0.02               # 2% stop loss (as fraction)
TAKE_PROFIT_PCT   = 0.04               # 4% take profit (as fraction)
RSI_PERIOD        = 14
RSI_OVERBOUGHT    = 70
RSI_OVERSOLD      = 30
ATR_PERIOD        = 14
MIN_ATR           = 0.01               # minimum ATR to filter low volatility

# ======== FETCH & PREPARE DATA =========
def fetch_and_prepare(symbol, timeframe, days, exchange_id, api_key):
    exchange = getattr(ccxt, exchange_id)({'apiKey': api_key, 'enableRateLimit': True})
    now = exchange.milliseconds()
    since = now - days * 24 * 3600 * 1000
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since)
    df = pd.DataFrame(ohlcv, columns=['Timestamp','Open','High','Low','Close','Volume'])
    df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df.set_index('Date', inplace=True)

    # Compute Typical Price (HLC3) and session VWAP
    df['HLC3'] = (df['High'] + df['Low'] + df['Close']) / 3
    df = df.groupby(df.index.date, group_keys=False).apply(
        lambda g: g.assign(
            VWAP=(g['HLC3'] * g['Volume']).cumsum() / g['Volume'].cumsum()
        )
    )

    # Compute RSI and ATR using pandas_ta
    df['RSI'] = ta.rsi(df['Close'], length=RSI_PERIOD)
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=ATR_PERIOD)

    return df.drop(columns=['Timestamp','HLC3'])

# ======== STRATEGY CLASS =========
class EnhancedVWAP(Strategy):
    def init(self):
        self.vwap = self.data.VWAP
        self.rsi  = self.data.RSI
        self.atr  = self.data.ATR
        # Plot VWAP
        self.I(lambda: self.vwap, name='VWAP')

    def next(self):
        price_open = self.data.Open[-1]
        price_close = self.data.Close[-1]
        vwap = self.vwap[-1]
        rsi  = self.rsi[-1]
        atr  = self.atr[-1]

        # Filter low volatility or missing values
        if atr < MIN_ATR or pd.isna(rsi):
            return

        # Prepare absolute SL/TP levels
        entry_price = price_open
        sl_long = entry_price * (1 - STOP_LOSS_PCT)
        tp_long = entry_price * (1 + TAKE_PROFIT_PCT)
        sl_short = entry_price * (1 + STOP_LOSS_PCT)
        tp_short = entry_price * (1 - TAKE_PROFIT_PCT)

        # Buy Condition: open < VWAP and RSI not overbought
        if not self.position and price_open < vwap and rsi < RSI_OVERBOUGHT:
            # Enter long with absolute SL/TP
            self.buy(size=LOT_SIZE, sl=sl_long, tp=tp_long)

        # Sell Condition: open > VWAP and RSI not oversold
        elif not self.position and price_open > vwap and rsi > RSI_OVERSOLD:
            # Enter short with absolute SL/TP
            self.sell(size=LOT_SIZE, sl=sl_short, tp=tp_short)

# ======== MAIN BACKTEST =========
def main():
    df = fetch_and_prepare(SYMBOL, TIMEFRAME, BACK_DAYS, EXCHANGE_ID, API_KEY)

    bt = Backtest(
        df,
        EnhancedVWAP,
        commission=COMMISSION_TAKER,
        trade_on_close=False,
        cash=10000,
        margin=1
    )
    stats = bt.run()

    # Print configuration
    print("=== Configuration ===")
    print(f"Symbol           : {SYMBOL}")
    print(f"Timeframe        : {TIMEFRAME}")
    print(f"Period           : {df.index[0].date()} to {df.index[-1].date()} ({BACK_DAYS}d)")
    print(f"Maker Comm.      : {COMMISSION_MAKER*100:.2f}%")
    print(f"Taker Comm.      : {COMMISSION_TAKER*100:.2f}%")
    print(f"Stop Loss        : {STOP_LOSS_PCT*100:.1f}%")
    print(f"Take Profit      : {TAKE_PROFIT_PCT*100:.1f}%")
    print(f"RSI Filters      : <{RSI_OVERBOUGHT}, >{RSI_OVERSOLD}")
    print(f"ATR Min          : {MIN_ATR}")
    print(f"Lot Size         : {LOT_SIZE}\n")

    # Print results
    print("=== Backtest Results ===")
    print(stats)

    # Plot chart with VWAP and trades
    bt.plot()

if __name__ == '__main__':
    main()
