
import ccxt
import pandas as pd
import pandas_ta as ta
import time
import matplotlib.pyplot as plt

# Configuration
EXCHANGE_ID = 'binance'                 # CCXT exchange ID
SYMBOL = 'SOL/USDT'                     # Trading pair
TIMEFRAME = '3m'                        # 5-minute candles

# Backtest interval: last 1 month
now_ms = int(time.time() * 1000)
one_month_ms = 30 * 24 * 60 * 60 * 1000
SINCE = now_ms - one_month_ms

# Capital & sizing
INITIAL_CAPITAL = 1000                # USD
POSITION_SIZE = 0.1                    # fraction of capital per trade

# Initialize exchange client with rate limit
exchange = getattr(ccxt, EXCHANGE_ID)({
    'enableRateLimit': True,
})

# Fetch historical OHLCV
ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, since=SINCE)
cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
df = pd.DataFrame(ohlcv, columns=cols)
df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('datetime', inplace=True)
df.drop(columns=['timestamp'], inplace=True)

# Calculate indicators
st = ta.supertrend(df['high'], df['low'], df['close'], length=20, multiplier=2)
macd = ta.macd(df['close'], fast=12, slow=26, signal=9)

df = df.join(st).join(macd)
df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])

# Backtest loop
signals = []
capital = INITIAL_CAPITAL
position = None       # None, 'long', or 'short'
entry_price = 0
qty = 0
capital_history = []

def macd_cross_down(prev_macd, prev_sig, macd, sig):
    return (prev_macd > prev_sig) and (macd < sig)

def macd_cross_up(prev_macd, prev_sig, macd, sig):
    return (prev_macd < prev_sig) and (macd > sig)

for i in range(1, len(df)):
    prev = df.iloc[i - 1]
    row = df.iloc[i]

    prev_trend = prev['SUPERTd_20_2.0']
    curr_trend = row['SUPERTd_20_2.0']
    prev_macd = prev['MACD_12_26_9']
    prev_sig  = prev['MACDs_12_26_9']
    curr_macd = row['MACD_12_26_9']
    curr_sig  = row['MACDs_12_26_9']
    price     = row['close']
    vwap      = row['vwap']

    # Short entry: VWAP > price, MACD cross down & ST gives sell
    if position is None and vwap > price \
       and macd_cross_down(prev_macd, prev_sig, curr_macd, curr_sig) \
       and (prev_trend >= 0 and curr_trend < 0):
        position = 'short'
        entry_price = price
        qty = (capital * POSITION_SIZE) / price
        signals.append((row.name, 'SHORT', price))

    # Long entry: VWAP < price, MACD cross up & ST gives buy
    if position is None and vwap < price \
       and macd_cross_up(prev_macd, prev_sig, curr_macd, curr_sig) \
       and (prev_trend <= 0 and curr_trend > 0):
        position = 'long'
        entry_price = price
        qty = (capital * POSITION_SIZE) / price
        signals.append((row.name, 'LONG', price))

    # Short exit: ST gives buy signal
    if position == 'short' and (prev_trend <= 0 and curr_trend > 0):
        exit_price = price
        pnl = (entry_price - exit_price) * qty
        capital += pnl
        signals.append((row.name, 'COVER', price, pnl))
        position = None

    # Long exit: ST gives sell signal
    if position == 'long' and (prev_trend >= 0 and curr_trend < 0):
        exit_price = price
        pnl = (exit_price - entry_price) * qty
        capital += pnl
        signals.append((row.name, 'SELL', price, pnl))
        position = None

    capital_history.append(capital)

# Results
signals_df = pd.DataFrame(signals, columns=['datetime', 'action', 'price', 'pnl']).set_index('datetime')
capital_series = pd.Series(capital_history, index=df.index[1:], name='capital')

print(f"Trades executed: {len(signals_df)}")
print(signals_df)
print(f"Final capital: {capital:.2f}")

# Equity curve
plt.figure(figsize=(10, 4))
capital_series.plot()
plt.title('Equity Curve')
plt.ylabel('Capital')
plt.show()

