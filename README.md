# 📈 Backtest Strategy

A **comprehensive Python-based backtesting suite** for algorithmic trading strategies. This project allows traders and developers to **evaluate and optimize trading strategies** using historical data, ensuring **data-driven decision-making** before live deployment.

---

## ✅ Features

* **🔄 Delta Trading Algorithm Backtester**
  A robust backtesting engine featuring:

  * Historical data fetching via **`ccxt`**
  * **Technical indicators**: Moving Averages, ATR
  * **Position sizing** with the **Kelly Criterion**
  * Trade simulation & **performance analytics**
  * **Interactive & static plots** for visual analysis

* **📊 Trend Following Strategy**
  Classic trend-following strategy using:

  * Multiple moving averages
  * Session high/low breakouts
  * Configurable **stop-loss** and **take-profit** (via `backtesting.py`)

* **📉 VWAP Trading Strategies**
  Includes **two variations**:

  * Basic VWAP entries & exits
  * Enhanced VWAP + **RSI & ATR filtering** for improved risk management

* **📌 Multi-Indicator Strategy**
  Combines **Supertrend, MACD, and VWAP** for a confluence-based approach.

* **📂 Historical Data Integration**
  Fetches accurate OHLCV data from multiple exchanges (**Delta, Binance, etc.**) using `ccxt`.

* **📈 Performance Analysis & Visualization**
  Generates detailed performance statistics: profitability, drawdown, trade statistics, and equity curves.

---

## 🗂️ Project Structure

```
.
├── 3cnfo.py                      # Multi-indicator strategy (Supertrend, MACD, VWAP)
├── backtest_results.png          # Example screenshot of backtest results
├── backtest.py                   # Delta Trading Algorithm backtester
├── imp.py                        # Enhanced VWAP strategy
├── Trend Following Backtest.py   # Trend Following strategy backtester
├── VWAPHLC3Strategy.html         # HTML plot output for VWAP strategy
├── VWAPStrategy.html             # HTML plot output for VWAP strategy
└── vwep.py                       # Basic VWAP strategy
```

---

## ⚙️ Prerequisites

Ensure you have the following installed:

* **Python 3.x**
* **pip** (Python package installer)

---

## 🛠️ Tech Stack

* **Python** – Core language
* **Pandas** – Data manipulation & analysis
* **NumPy** – Numerical computations
* **CCXT** – Cryptocurrency exchange data fetching
* **Matplotlib** – Static plots
* **`backtesting.py`** – Backtesting framework
* **`pandas_ta`** – Technical indicators

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your_username/backtest-strategy.git
cd backtest-strategy
```

*(Replace `your_username/backtest-strategy.git` with your actual repository URL)*

### 2️⃣ Install Dependencies

```bash
pip install pandas numpy ccxt matplotlib backtesting pandas_ta
```

---

## ▶️ Running Backtests

**Run Delta Trading Algorithm Backtest:**

```bash
python backtest.py
```

**Run Trend Following Backtest:**

```bash
python "Trend Following Backtest.py"
```

Each script will output **performance statistics** and generate **plots/HTML files** (e.g., `backtest_results.png` or `backtest.html`).

---

## ⚡ Configuration

You can customize:

* **SYMBOL** (e.g., BTC/USDT)
* **TIMEFRAME** (e.g., 1h, 4h)
* **EXCHANGE\_ID** (e.g., binance, kraken)
* **LOT\_SIZE**

### 🔑 Environment Variables for API Keys

**Linux/macOS:**

```bash
export DELTA_API_KEY="your_actual_delta_api_key"
```

**Windows:**

```cmd
set DELTA_API_KEY="your_actual_delta_api_key"
```

Access in Python:

```python
import os
API_KEY = os.getenv('DELTA_API_KEY', 'YOUR_API_KEY_DEFAULT_IF_NOT_SET')
```

## ❓ FAQ

**Q: How do I change the cryptocurrency symbol or timeframe?**
A: Edit configuration variables (`SYMBOL`, `TIMEFRAME`, etc.) at the top of each Python script.

**Q: Can I use different exchanges?**
A: Yes. Change `EXCHANGE_ID` to supported exchanges in `ccxt` (e.g., binance, kraken, bybit).

**Q: Why multiple backtesting scripts?**
A: Each represents a unique trading strategy or backtesting approach.

**Q: How to interpret results?**
A: Check **equity curves**, **drawdown**, and **profitability metrics** in generated plots and console outputs.
