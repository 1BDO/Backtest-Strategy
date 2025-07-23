import ccxt
import pandas as pd
import pandas_ta as ta
from backtesting import Backtest, Strategy

# ======== CONFIGURATION (modifiable) =========
EXCHANGE_ID      = 'delta'             # ccxt exchange id ('delta', 'binance', etc.)
API_KEY          = 'YOUR_API_KEY'      # optional
SYMBOL           = 'XRP/USDT'          # symbol to backtest
TIMEFRAME        = '15m'                # intraday timeframe e.g. '5m'
BACK_DAYS        = 15                  # days to backtest
COMMISSION       = 0.0005              # taker commission 0.05%
LOT_SIZE         = 1                   # position size multiplier

# Moving averages periods
MA_FAST          = 20
MA_MED           = 50
MA_SLOW          = 200

# Session breakout settings
SESSION_BARS     = 6                   # bars after daily open (e.g., 6 x 5m = 30m)

# Risk management
SL_PCT           = 0.005               # fixed SL 0.5% or use session low/high
TP_FACTOR        = 2.0                 # take-profit at TP_FACTOR x risk

# ======== FETCH & PREPARE DATA =========
def fetch_data(symbol, timeframe, days, exchange_id, api_key):
    exchange = getattr(ccxt, exchange_id)({'apiKey': api_key, 'enableRateLimit': True})
    now = exchange.milliseconds()
    since = now - days * 24 * 3600 * 1000
    data = exchange.fetch_ohlcv(symbol, timeframe, since=since)
    df = pd.DataFrame(data, columns=['ts','Open','High','Low','Close','Volume'])
    df['Date'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('Date', inplace=True)
    df = df[['Open','High','Low','Close','Volume']]
    return df

# ======== PREPROCESSING =========
def prepare_indicators(df):
    # Compute MAs
    df[f'MA{MA_FAST}'] = ta.sma(df['Close'], length=MA_FAST)
    df[f'MA{MA_MED}']  = ta.sma(df['Close'], length=MA_MED)
    df[f'MA{MA_SLOW}'] = ta.sma(df['Close'], length=MA_SLOW)

    # Session high/low breakout: get first SESSION_BARS bars high/low per day
    df['date_only'] = df.index.date
    highs = df.groupby('date_only', group_keys=False).apply(lambda g: g['High'].iloc[:SESSION_BARS].max())
    lows  = df.groupby('date_only', group_keys=False).apply(lambda g: g['Low'].iloc[:SESSION_BARS].min())
    df['session_high'] = df['date_only'].map(highs)
    df['session_low']  = df['date_only'].map(lows)
    df.drop(columns=['date_only'], inplace=True)
    return df.dropna()

# ======== STRATEGY DEFINITION =========
class TrendFollowing(Strategy):
    def init(self):
        self.ma_fast = self.data[f'MA{MA_FAST}']
        self.ma_med  = self.data[f'MA{MA_MED}']
        self.ma_slow = self.data[f'MA{MA_SLOW}']
        self.sh = self.data.session_high
        self.sl = self.data.session_low

    def next(self):
        price_open = self.data.Open[-1]
        price_high = self.data.High[-1]
        price_low  = self.data.Low[-1]

        # Determine trend bias
        uptrend   = self.ma_fast[-1] > self.ma_med[-1] > self.ma_slow[-1]
        downtrend = self.ma_fast[-1] < self.ma_med[-1] < self.ma_slow[-1]

        # Session breakout levels
        breakout_high = self.sh[-1]
        breakout_low  = self.sl[-1]

        # Debugging output
        print(f"Price High: {price_high}, Breakout High: {breakout_high}, Uptrend: {uptrend}")
        print(f"Price Low: {price_low}, Breakout Low: {breakout_low}, Downtrend: {downtrend}")

        # Long entry: breakout above session_high in uptrend
        if not self.position and uptrend and price_high > breakout_high:
            entry_price = breakout_high
            stop_price  = breakout_low
            risk        = entry_price - stop_price
            tp_price    = entry_price + abs(risk) * TP_FACTOR  # Ensure tp_price is greater than entry_price
            if stop_price < entry_price < tp_price:  # Validate order parameters
                self.buy(size=LOT_SIZE, sl=stop_price, tp=tp_price)

        # Short entry: breakout below session_low in downtrend
        elif not self.position and downtrend and price_low < breakout_low:
            entry_price = breakout_low
            stop_price  = breakout_high
            risk        = stop_price - entry_price
            tp_price    = entry_price - abs(risk) * TP_FACTOR  # Ensure tp_price is less than entry_price
            if tp_price < entry_price < stop_price:  # Validate order parameters
                self.sell(size=LOT_SIZE, sl=stop_price, tp=tp_price)

# ======== MAIN BACKTEST =========
def main():
    df = fetch_data(SYMBOL, TIMEFRAME, BACK_DAYS, EXCHANGE_ID, API_KEY)
    df = prepare_indicators(df)

    bt = Backtest(
        df,
        TrendFollowing,
        commission=COMMISSION,
        trade_on_close=False,
        cash=100,
        margin=1
    )

    stats = bt.run()
    print("=== Backtest Summary ===")
    print(stats)
    bt.plot()

if __name__ == '__main__':
    main()
