import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import threading
import time

class Position:
    """Represents a trading position"""
    
    def __init__(self, symbol: str, quantity: int, entry_price: float, 
                 entry_time: datetime, position_type: str = 'long'):
        self.symbol = symbol
        self.quantity = quantity
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.position_type = position_type  # 'long' or 'short'
        self.current_price = entry_price
        self.last_update = entry_time
        
    def update_price(self, new_price: float):
        """Update current price"""
        self.current_price = new_price
        self.last_update = datetime.now()
    
    def get_market_value(self) -> float:
        """Get current market value"""
        return self.quantity * self.current_price
    
    def get_unrealized_pnl(self) -> float:
        """Get unrealized P&L"""
        if self.position_type == 'long':
            return (self.current_price - self.entry_price) * self.quantity
        else:  # short position
            return (self.entry_price - self.current_price) * self.quantity
    
    def get_unrealized_pnl_percent(self) -> float:
        """Get unrealized P&L as percentage"""
        if self.entry_price == 0:
            return 0.0
        return (self.get_unrealized_pnl() / (self.entry_price * abs(self.quantity))) * 100

class PortfolioManager:
    """Manages portfolio positions and performance tracking"""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.positions = {}  # symbol -> Position
        self.cash_balance = config.get('initial_capital', 100000)
        self.initial_capital = self.cash_balance
        self.trade_history = []
        self.daily_pnl_history = []
        self.portfolio_history = []
        self.last_portfolio_update = datetime.now()
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_commission = 0.0
        
        # Lock for thread safety
        self._lock = threading.Lock()
    
    def add_position(self, symbol: str, quantity: int, price: float, 
                    position_type: str = 'long') -> bool:
        """Add a new position or update existing position"""
        try:
            with self._lock:
                if symbol in self.positions:
                    # Update existing position
                    existing_pos = self.positions[symbol]
                    
                    # Calculate new average entry price
                    total_quantity = existing_pos.quantity + quantity
                    if total_quantity != 0:
                        new_avg_price = (
                            (existing_pos.entry_price * existing_pos.quantity) + 
                            (price * quantity)
                        ) / total_quantity
                        
                        existing_pos.quantity = total_quantity
                        existing_pos.entry_price = new_avg_price
                        existing_pos.last_update = datetime.now()
                        
                        if total_quantity == 0:
                            # Position closed
                            del self.positions[symbol]
                    else:
                        # Position reversed
                        self.positions[symbol] = Position(
                            symbol, quantity, price, datetime.now(), position_type
                        )
                else:
                    # New position
                    self.positions[symbol] = Position(
                        symbol, quantity, price, datetime.now(), position_type
                    )
                
                # Update cash balance
                trade_value = abs(quantity) * price
                commission = self._calculate_commission(trade_value)
                
                if quantity > 0:  # Buy
                    self.cash_balance -= (trade_value + commission)
                else:  # Sell
                    self.cash_balance += (trade_value - commission)
                
                self.total_commission += commission
                self.total_trades += 1
                
                # Record trade
                self._record_trade(symbol, quantity, price, commission)
                
                self.logger.info(f"Position updated: {symbol} {quantity} @ ${price:.2f}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error adding position for {symbol}: {str(e)}")
            return False
    
    def close_position(self, symbol: str, price: Optional[float] = None) -> bool:
        """Close a position completely"""
        try:
            with self._lock:
                if symbol not in self.positions:
                    self.logger.warning(f"No position found for {symbol}")
                    return False
                
                position = self.positions[symbol]
                close_price = price or position.current_price
                
                # Calculate P&L
                pnl = position.get_unrealized_pnl()
                if price:
                    position.update_price(price)
                    pnl = position.get_unrealized_pnl()
                
                # Close position (sell all shares)
                trade_value = abs(position.quantity) * close_price
                commission = self._calculate_commission(trade_value)
                
                self.cash_balance += trade_value - commission
                self.total_commission += commission
                
                # Record trade
                self._record_trade(symbol, -position.quantity, close_price, commission, pnl)
                
                # Track winning/losing trades
                if pnl > 0:
                    self.winning_trades += 1
                
                # Remove position
                del self.positions[symbol]
                
                self.logger.info(f"Position closed: {symbol} P&L: ${pnl:.2f}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error closing position for {symbol}: {str(e)}")
            return False
    
    def update_positions(self):
        """Update all position prices with current market data"""
        try:
            from trading_system.data_processor import DataProcessor
            data_processor = DataProcessor(self.config, self.logger)
            
            with self._lock:
                for symbol, position in self.positions.items():
                    try:
                        # Get current price
                        data = data_processor.get_market_data(symbol, period="1d", interval="1m")
                        if data is not None and not data.empty:
                            current_price = data['Close'].iloc[-1]
                            position.update_price(current_price)
                    except Exception as e:
                        self.logger.error(f"Error updating price for {symbol}: {str(e)}")
                        continue
                
                # Update portfolio history
                self._update_portfolio_history()
                
        except Exception as e:
            self.logger.error(f"Error updating positions: {str(e)}")
    
    def _calculate_commission(self, trade_value: float) -> float:
        """Calculate trading commission"""
        commission_rate = self.config.get('commission_rate', 0.001)  # 0.1%
        min_commission = self.config.get('min_commission', 1.0)
        return max(min_commission, trade_value * commission_rate)
    
    def _record_trade(self, symbol: str, quantity: int, price: float, 
                     commission: float, realized_pnl: float = 0):
        """Record a trade in history"""
        trade_record = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'trade_value': abs(quantity) * price,
            'commission': commission,
            'realized_pnl': realized_pnl,
            'action': 'BUY' if quantity > 0 else 'SELL'
        }
        
        self.trade_history.append(trade_record)
        
        # Keep only last 1000 trades
        if len(self.trade_history) > 1000:
            self.trade_history = self.trade_history[-1000:]
    
    def _update_portfolio_history(self):
        """Update portfolio value history"""
        try:
            current_value = self.get_total_portfolio_value()
            portfolio_record = {
                'timestamp': datetime.now(),
                'total_value': current_value,
                'cash': self.cash_balance,
                'positions_value': current_value - self.cash_balance,
                'unrealized_pnl': self.get_total_unrealized_pnl(),
                'num_positions': len(self.positions)
            }
            
            self.portfolio_history.append(portfolio_record)
            
            # Keep only last 10000 records (about 7 days of minute data)
            if len(self.portfolio_history) > 10000:
                self.portfolio_history = self.portfolio_history[-10000:]
                
        except Exception as e:
            self.logger.error(f"Error updating portfolio history: {str(e)}")
    
    def get_total_portfolio_value(self) -> float:
        """Get total portfolio value (cash + positions)"""
        try:
            with self._lock:
                positions_value = sum(pos.get_market_value() for pos in self.positions.values())
                return self.cash_balance + positions_value
        except Exception as e:
            self.logger.error(f"Error calculating portfolio value: {str(e)}")
            return self.cash_balance
    
    def get_total_unrealized_pnl(self) -> float:
        """Get total unrealized P&L"""
        try:
            with self._lock:
                return sum(pos.get_unrealized_pnl() for pos in self.positions.values())
        except Exception as e:
            self.logger.error(f"Error calculating unrealized P&L: {str(e)}")
            return 0.0
    
    def get_daily_pnl(self) -> float:
        """Get today's P&L"""
        try:
            if not self.portfolio_history:
                return 0.0
            
            # Find start of day value
            today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            
            start_value = None
            for record in reversed(self.portfolio_history):
                if record['timestamp'] <= today_start:
                    start_value = record['total_value']
                    break
            
            if start_value is None:
                start_value = self.initial_capital
            
            current_value = self.get_total_portfolio_value()
            return current_value - start_value
            
        except Exception as e:
            self.logger.error(f"Error calculating daily P&L: {str(e)}")
            return 0.0
    
    def get_portfolio_summary(self) -> Dict:
        """Get comprehensive portfolio summary"""
        try:
            total_value = self.get_total_portfolio_value()
            daily_pnl = self.get_daily_pnl()
            unrealized_pnl = self.get_total_unrealized_pnl()
            
            # Calculate realized P&L from trades
            realized_pnl = sum(trade.get('realized_pnl', 0) for trade in self.trade_history)
            
            return {
                'total_value': total_value,
                'cash': self.cash_balance,
                'positions_value': total_value - self.cash_balance,
                'daily_pnl': daily_pnl,
                'daily_pnl_pct': (daily_pnl / total_value * 100) if total_value > 0 else 0,
                'unrealized_pnl': unrealized_pnl,
                'realized_pnl': realized_pnl,
                'total_return': ((total_value - self.initial_capital) / self.initial_capital * 100) if self.initial_capital > 0 else 0,
                'open_positions': len(self.positions),
                'total_trades': self.total_trades,
                'win_rate': (self.winning_trades / max(1, self.total_trades)) * 100,
                'total_commission': self.total_commission
            }
            
        except Exception as e:
            self.logger.error(f"Error getting portfolio summary: {str(e)}")
            return {}
    
    def get_all_positions(self) -> Dict:
        """Get all current positions"""
        try:
            with self._lock:
                positions_dict = {}
                for symbol, position in self.positions.items():
                    positions_dict[symbol] = {
                        'quantity': position.quantity,
                        'entry_price': position.entry_price,
                        'current_price': position.current_price,
                        'market_value': position.get_market_value(),
                        'unrealized_pnl': position.get_unrealized_pnl(),
                        'unrealized_pnl_pct': position.get_unrealized_pnl_percent(),
                        'entry_time': position.entry_time,
                        'position_type': position.position_type
                    }
                return positions_dict
        except Exception as e:
            self.logger.error(f"Error getting positions: {str(e)}")
            return {}
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get specific position details"""
        try:
            with self._lock:
                if symbol in self.positions:
                    position = self.positions[symbol]
                    return {
                        'symbol': symbol,
                        'quantity': position.quantity,
                        'entry_price': position.entry_price,
                        'current_price': position.current_price,
                        'market_value': position.get_market_value(),
                        'unrealized_pnl': position.get_unrealized_pnl(),
                        'unrealized_pnl_pct': position.get_unrealized_pnl_percent(),
                        'entry_time': position.entry_time,
                        'position_type': position.position_type
                    }
                return None
        except Exception as e:
            self.logger.error(f"Error getting position for {symbol}: {str(e)}")
            return None
    
    def get_performance_history(self) -> Optional[pd.DataFrame]:
        """Get portfolio performance history as DataFrame"""
        try:
            if not self.portfolio_history:
                return None
            
            df = pd.DataFrame(self.portfolio_history)
            df['date'] = pd.to_datetime(df['timestamp'])
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting performance history: {str(e)}")
            return None
    
    def get_trade_history(self, days: int = 30) -> List[Dict]:
        """Get recent trade history"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_trades = [
                trade for trade in self.trade_history 
                if trade['timestamp'] > cutoff_date
            ]
            return recent_trades
        except Exception as e:
            self.logger.error(f"Error getting trade history: {str(e)}")
            return []
    
    def calculate_portfolio_metrics(self) -> Dict:
        """Calculate advanced portfolio metrics"""
        try:
            if not self.portfolio_history:
                return {}
            
            df = pd.DataFrame(self.portfolio_history)
            returns = df['total_value'].pct_change().dropna()
            
            if len(returns) < 2:
                return {}
            
            # Calculate metrics
            total_return = (df['total_value'].iloc[-1] / df['total_value'].iloc[0] - 1) * 100
            volatility = returns.std() * np.sqrt(252 * 24 * 60)  # Annualized (assuming minute data)
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252 * 24 * 60) if returns.std() > 0 else 0
            
            # Max drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = cumulative / running_max - 1
            max_drawdown = drawdown.min() * 100
            
            return {
                'total_return': total_return,
                'annualized_volatility': volatility * 100,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': abs(max_drawdown),
                'total_trades': self.total_trades,
                'win_rate': (self.winning_trades / max(1, self.total_trades)) * 100,
                'avg_trade_value': np.mean([trade['trade_value'] for trade in self.trade_history]) if self.trade_history else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio metrics: {str(e)}")
            return {}
    
    def rebalance_portfolio(self, target_weights: Dict[str, float]) -> bool:
        """Rebalance portfolio to target weights"""
        try:
            total_value = self.get_total_portfolio_value()
            current_positions = self.get_all_positions()
            
            self.logger.info("Starting portfolio rebalancing...")
            
            for symbol, target_weight in target_weights.items():
                target_value = total_value * target_weight
                current_value = current_positions.get(symbol, {}).get('market_value', 0)
                
                # Calculate required trade
                value_diff = target_value - current_value
                
                if abs(value_diff) > 100:  # Only trade if difference > $100
                    # Get current price for the symbol
                    from trading_system.data_processor import DataProcessor
                    data_processor = DataProcessor(self.config, self.logger)
                    data = data_processor.get_market_data(symbol, period="1d", interval="1m")
                    
                    if data is not None and not data.empty:
                        current_price = data['Close'].iloc[-1]
                        shares_to_trade = int(value_diff / current_price)
                        
                        if shares_to_trade != 0:
                            self.add_position(symbol, shares_to_trade, current_price)
                            self.logger.info(f"Rebalanced {symbol}: {shares_to_trade} shares")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error rebalancing portfolio: {str(e)}")
            return False
