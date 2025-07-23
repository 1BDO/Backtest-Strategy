# backtest.py
# Backtesting script for the Delta Trading Algorithm

import pandas as pd
import numpy as np
import ccxt
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('backtest')

class DeltaStrategyBacktester:
    def __init__(self, symbol='BTC/USD', timeframe='1d', initial_balance=1000):
        self.symbol = symbol
        self.timeframe = timeframe
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.risk_reward_ratio = 3
        self.kelly_fraction = 0.5
        self.win_probability = 0.6
        self.position = None
        self.trades = []
        self.equity_curve = []
        
    def fetch_historical_data(self, days=365):
        """Fetch historical data using CCXT"""
        try:
            logger.info(f"Fetching {days} days of historical data for {self.symbol}")
            exchange = ccxt.delta()
            since = exchange.milliseconds() - (days * 24 * 60 * 60 * 1000)
            ohlcv = exchange.fetch_ohlcv(self.symbol, self.timeframe, since=since)
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"Fetched {len(df)} records of historical data")
            return df
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            raise
    
    def calculate_indicators(self, df):
        """Calculate technical indicators"""
        logger.info("Calculating technical indicators")
        
        # Calculate moving averages
        df['ma_200'] = df['close'].rolling(window=200).mean()
        df['ma_50'] = df['close'].rolling(window=50).mean()
        
        # Calculate ATR
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(window=14).mean()
        
        # Additional indicators for visualization
        df['trend'] = np.where(df['close'] > df['ma_200'], 'uptrend', 'downtrend')
        
        return df.dropna()
    
    def determine_entry_signals(self, df):
        """Determine entry signals based on strategy rules"""
        # Initialize signal columns
        df['entry_signal'] = None
        df['stop_loss'] = None
        df['take_profit'] = None
        
        # Loop through data to determine entry points and exit levels
        for i in range(1, len(df)):
            current_price = df['close'].iloc[i]
            ma_200 = df['ma_200'].iloc[i]
            ma_50 = df['ma_50'].iloc[i]
            atr = df['atr'].iloc[i]
            
            # Uptrend: buy when price dips below 50-day MA but stays above 200-day MA
            if current_price > ma_200 and current_price < ma_50:
                df['entry_signal'].iloc[i] = 'buy'
                df['stop_loss'].iloc[i] = current_price - (2 * atr)
                df['take_profit'].iloc[i] = current_price + (self.risk_reward_ratio * 2 * atr)
            
            # Downtrend: sell when price rallies above 50-day MA but stays below 200-day MA
            elif current_price < ma_200 and current_price > ma_50:
                df['entry_signal'].iloc[i] = 'sell'
                df['stop_loss'].iloc[i] = current_price + (2 * atr)
                df['take_profit'].iloc[i] = current_price - (self.risk_reward_ratio * 2 * atr)
        
        return df
    
    def calculate_position_size(self, price, atr, balance):
        """Calculate position size using Kelly Criterion"""
        # Kelly formula: f = (bp - q) / b
        p = self.win_probability
        q = 1 - p
        b = self.risk_reward_ratio
        
        # Calculate Kelly percentage and apply safety factor
        kelly_percentage = (b * p - q) / b * self.kelly_fraction
        
        # Calculate stop loss distance (2 ATR)
        stop_loss_distance = 2 * atr
        
        # Calculate risk amount for this trade
        risk_per_trade = kelly_percentage * balance
        
        # Calculate position size
        position_size = risk_per_trade / stop_loss_distance
        
        return position_size
    
    def simulate_trades(self, df):
        """Simulate trades based on entry signals"""
        balance = self.initial_balance
        equity_curve = [balance]
        trades = []
        position = None
        
        for i in range(1, len(df) - 1):
            date = df.index[i]
            current_price = df['close'].iloc[i]
            next_high = df['high'].iloc[i+1]
            next_low = df['low'].iloc[i+1]
            next_close = df['close'].iloc[i+1]
            
            # Check if we have an open position
            if position:
                # Check if stop loss was hit
                if (position['side'] == 'buy' and next_low <= position['stop_loss']) or \
                   (position['side'] == 'sell' and next_high >= position['stop_loss']):
                    # Stop loss hit
                    pnl = position['size'] * (position['stop_loss'] - position['entry_price']) \
                          if position['side'] == 'buy' else \
                          position['size'] * (position['entry_price'] - position['stop_loss'])
                    
                    balance += pnl
                    trades.append({
                        'entry_date': position['entry_date'],
                        'exit_date': date,
                        'side': position['side'],
                        'entry_price': position['entry_price'],
                        'exit_price': position['stop_loss'],
                        'size': position['size'],
                        'pnl': pnl,
                        'exit_type': 'stop_loss',
                        'balance': balance
                    })
                    position = None
                
                # Check if take profit was hit
                elif (position['side'] == 'buy' and next_high >= position['take_profit']) or \
                     (position['side'] == 'sell' and next_low <= position['take_profit']):
                    # Take profit hit
                    pnl = position['size'] * (position['take_profit'] - position['entry_price']) \
                          if position['side'] == 'buy' else \
                          position['size'] * (position['entry_price'] - position['take_profit'])
                    
                    balance += pnl
                    trades.append({
                        'entry_date': position['entry_date'],
                        'exit_date': date,
                        'side': position['side'],
                        'entry_price': position['entry_price'],
                        'exit_price': position['take_profit'],
                        'size': position['size'],
                        'pnl': pnl,
                        'exit_type': 'take_profit',
                        'balance': balance
                    })
                    position = None
            
            # Check for new entry signals if we don't have an open position
            if not position and df['entry_signal'].iloc[i]:
                entry_side = df['entry_signal'].iloc[i]
                entry_price = current_price
                stop_loss = df['stop_loss'].iloc[i]
                take_profit = df['take_profit'].iloc[i]
                atr = df['atr'].iloc[i]
                
                # Calculate position size
                size = self.calculate_position_size(entry_price, atr, balance)
                
                # Open new position
                position = {
                    'entry_date': date,
                    'side': entry_side,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'size': size
                }
            
            # Record equity at the end of each day
            equity_curve.append(balance)
        
        return trades, equity_curve
    
    def run_backtest(self):
        """Run complete backtest"""
        # Fetch historical data
        df = self.fetch_historical_data()
        
        # Calculate indicators
        df = self.calculate_indicators(df)
        
        # Determine entry signals
        df = self.determine_entry_signals(df)
        
        # Simulate trades
        trades, equity_curve = self.simulate_trades(df)
        
        # Calculate performance metrics
        performance = self.calculate_performance(trades, equity_curve)
        
        # Store results
        self.df = df
        self.trades = trades
        self.equity_curve = equity_curve
        self.performance = performance
        
        return performance
    
    def calculate_performance(self, trades, equity_curve):
        """Calculate performance metrics"""
        if not trades:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "total_return": 0,
                "max_drawdown": 0,
                "sharpe_ratio": 0
            }
        
        # Basic metrics
        total_trades = len(trades)
        winning_trades = sum(1 for trade in trades if trade['pnl'] > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Profit metrics
        total_profit = sum(trade['pnl'] for trade in trades if trade['pnl'] > 0)
        total_loss = sum(abs(trade['pnl']) for trade in trades if trade['pnl'] < 0)
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Calculate returns
        initial_equity = self.initial_balance
        final_equity = equity_curve[-1]
        total_return = (final_equity - initial_equity) / initial_equity * 100
        
        # Calculate drawdown
        max_equity = initial_equity
        max_drawdown = 0
        
        for equity in equity_curve:
            max_equity = max(max_equity, equity)
            drawdown = (max_equity - equity) / max_equity * 100
            max_drawdown = max(max_drawdown, drawdown)
        
        # Calculate Sharpe ratio (simplified)
        daily_returns = [
            (equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1] 
            for i in range(1, len(equity_curve))
        ]
        avg_return = np.mean(daily_returns)
        std_return = np.std(daily_returns)
        sharpe_ratio = avg_return / std_return * np.sqrt(252) if std_return > 0 else 0
        
        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": total_trades - winning_trades,
            "win_rate": win_rate * 100,
            "profit_factor": profit_factor,
            "initial_balance": initial_equity,
            "final_balance": final_equity,
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio
        }
    
    def plot_results(self):
        """Plot backtest results"""
        if not hasattr(self, 'df') or not hasattr(self, 'equity_curve'):
            logger.error("Cannot plot results: No backtest data available")
            return
        
        # Create figure with multiple subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
        
        # Price and MA chart
        ax1.plot(self.df.index, self.df['close'], label='Price', color='black', alpha=0.5)
        ax1.plot(self.df.index, self.df['ma_200'], label='200-day MA', color='red')
        ax1.plot(self.df.index, self.df['ma_50'], label='50-day MA', color='blue')
        
        # Plot entry and exit points
        for trade in self.trades:
            # Entry point
            if trade['side'] == 'buy':
                ax1.scatter(trade['entry_date'], trade['entry_price'], color='green', marker='^', s=100)
                ax1.scatter(trade['exit_date'], trade['exit_price'], 
                           color='red' if trade['exit_type'] == 'stop_loss' else 'blue', 
                           marker='v', s=100)
            else:
                ax1.scatter(trade['entry_date'], trade['entry_price'], color='red', marker='v', s=100)
                ax1.scatter(trade['exit_date'], trade['exit_price'], 
                           color='red' if trade['exit_type'] == 'stop_loss' else 'blue', 
                           marker='^', s=100)
        
        ax1.set_title('Price Chart with Moving Averages and Trade Signals')
        ax1.set_ylabel('Price')
        ax1.grid(True)
        ax1.legend()
        
        # Equity curve
        ax2.plot(self.df.index[:len(self.equity_curve)], self.equity_curve, label='Equity Curve', color='green')
        ax2.set_title('Equity Curve')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Equity')
        ax2.grid(True)
        
        # Add performance metrics as text
        if hasattr(self, 'performance'):
            perf_text = (
                f"Total Return: {self.performance['total_return']:.2f}%\n"
                f"Win Rate: {self.performance['win_rate']:.2f}%\n"
                f"Profit Factor: {self.performance['profit_factor']:.2f}\n"
                f"Max Drawdown: {self.performance['max_drawdown']:.2f}%\n"
                f"Sharpe Ratio: {self.performance['sharpe_ratio']:.2f}\n"
                f"Total Trades: {self.performance['total_trades']}"
            )
            ax2.annotate(perf_text, xy=(0.02, 0.02), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.7))
        
        plt.tight_layout()
        plt.savefig('backtest_results.png')
        logger.info("Results plotted and saved to 'backtest_results.png'")
        plt.show()
        
        # Print trade list
        self.print_trade_summary()
    
    def print_trade_summary(self):
        """Print summary of trades"""
        if not self.trades:
            print("No trades were executed during the backtest")
            return
        
        print("\n==== Trade Summary ====")
        print(f"Total Trades: {self.performance['total_trades']}")
        print(f"Winning Trades: {self.performance['winning_trades']} ({self.performance['win_rate']:.2f}%)")
        print(f"Losing Trades: {self.performance['losing_trades']}")
        print(f"Profit Factor: {self.performance['profit_factor']:.2f}")
        print(f"Initial Balance: ${self.performance['initial_balance']:.2f}")
        print(f"Final Balance: ${self.performance['final_balance']:.2f}")
        print(f"Total Return: {self.performance['total_return']:.2f}%")
        print(f"Max Drawdown: {self.performance['max_drawdown']:.2f}%")
        print(f"Sharpe Ratio: {self.performance['sharpe_ratio']:.2f}")
        
        print("\n==== Individual Trades ====")
        for i, trade in enumerate(self.trades, 1):
            print(f"Trade #{i}:")
            print(f"  Entry Date: {trade['entry_date']}")
            print(f"  Exit Date: {trade['exit_date']}")
            print(f"  Side: {trade['side']}")
            print(f"  Entry Price: ${trade['entry_price']:.2f}")
            print(f"  Exit Price: ${trade['exit_price']:.2f}")
            print(f"  Exit Type: {trade['exit_type']}")
            print(f"  P&L: ${trade['pnl']:.2f}")
            print(f"  Balance After Trade: ${trade['balance']:.2f}")
            print()

# Run backtest
if __name__ == "__main__":
    try:
        backtest = DeltaStrategyBacktester(symbol='BTC/USD', timeframe='1d', initial_balance=1000)
        performance = backtest.run_backtest()
        
        print("Backtest completed successfully!")
        print(f"Total Return: {performance['total_return']:.2f}%")
        print(f"Win Rate: {performance['win_rate']:.2f}%")
        print(f"Profit Factor: {performance['profit_factor']:.2f}")
        
        # Plot results
        backtest.plot_results()
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        import traceback
        traceback.print_exc()