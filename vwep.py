import ccxt
import pandas as pd
from backtesting import Backtest, Strategy

# ======== CONFIGURATION (modifiable) =========
EXCHANGE_ID       = 'delta'            # CCXT exchange id ('delta', 'binance', etc.)
API_KEY           = 'YOUR_API_KEY'     # optional for public data
SYMBOL            = 'XRP/USDT'         # symbol to backtest
TIMEFRAME         = '1d'               # timeframe: '1d', '1h', '1m', etc.
BACK_DAYS         = 7                 # lookback period in days
COMMISSION_MAKER  = 0.0002             # maker commission (0.02%)
COMMISSION_TAKER  = 0.0005             # taker commission (0.05%)
LOT_SIZE          = 1                # size per trade

# ======== FETCH VIA CCXT & PREPARE DATA =========
def fetch_and_prepare(symbol, timeframe, back_days, exchange_id, api_key):
    exchange = getattr(ccxt, exchange_id)({
        'apiKey': api_key,
        'enableRateLimit': True,
    })
    now_ms = exchange.milliseconds()
    since_ms = now_ms - back_days * 24 * 60 * 60 * 1000

    # Fetch OHLCV
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since_ms)
    df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df.set_index('Date', inplace=True)

    # Compute Typical Price (HLC3)
    df['HLC3'] = (df['High'] + df['Low'] + df['Close']) / 3

    # VWAP per session/day: reset at each new date
    def session_vwap(group):
        cum_vol = group['Volume'].cumsum()
        cum_pv = (group['HLC3'] * group['Volume']).cumsum()
        group['VWAP'] = cum_pv / cum_vol
        return group

    df = df.groupby(df.index.date, group_keys=False).apply(session_vwap)

    return df[['Open', 'High', 'Low', 'Close', 'Volume', 'VWAP']]

# ======== STRATEGY DEFINITION =========
class VWAPStrategy(Strategy):
    def init(self):
        # Reference VWAP column and plot it
        self.vwap = self.data.VWAP
        self.I(lambda: self.vwap, name='VWAP')

    def next(self):
        open_p = self.data.Open[-1]
        vwap_p = self.vwap[-1]
        # All entries/exits as taker orders
        if open_p < vwap_p:
            self.buy(size=LOT_SIZE)
        else:
            self.sell(size=LOT_SIZE)
        # Exit at close of same bar
        if self.position.is_long:
            self.sell(size=LOT_SIZE)
        elif self.position.is_short:
            self.buy(size=LOT_SIZE)

# ======== MAIN BACKTEST =========
def main():
    df = fetch_and_prepare(SYMBOL, TIMEFRAME, BACK_DAYS, EXCHANGE_ID, API_KEY)

    # Use taker commission since both entry and exit are market orders
    bt = Backtest(
        df,
        VWAPStrategy,
        commission=COMMISSION_TAKER,
        trade_on_close=True,
        cash=1000,
        margin=1
    )
    stats = bt.run()

    # Print configuration & results
    print("=== Backtest Configuration ===")
    print(f"Symbol            : {SYMBOL}")
    print(f"Timeframe         : {TIMEFRAME}")
    print(f"Period            : {df.index[0].date()} to {df.index[-1].date()} ({BACK_DAYS} days)")
    print(f"Maker Commission  : {COMMISSION_MAKER * 100:.2f}%")
    print(f"Taker Commission  : {COMMISSION_TAKER * 100:.2f}%")
    print(f"Lot Size          : {LOT_SIZE}\n")

    print("=== Backtest Results ===")
    print(stats)

    # Plot price, VWAP, and trades
    bt.plot()

if __name__ == '__main__':
    main()
