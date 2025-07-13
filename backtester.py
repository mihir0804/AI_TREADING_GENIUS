import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import threading
import time

class Backtester:
    """Comprehensive backtesting engine for strategy validation"""
    
    def __init__(self, config, logger, data_processor):
        self.config = config
        self.logger = logger
        self.data_processor = data_processor
        self.backtest_results = {}
        
    def run_backtest(self, strategy: str, start_date: datetime, end_date: datetime, 
                    initial_capital: float = 100000) -> Dict:
        """Run comprehensive backtest for a trading strategy"""
        try:
            self.logger.info(f"Starting backtest for {strategy} from {start_date} to {end_date}")
            
            # Initialize backtest state
            backtest_state = {
                'capital': initial_capital,
                'positions': {},
                'trades': [],
                'equity_curve': [],
                'daily_returns': [],
                'commission_paid': 0.0
            }
            
            # Get historical data for all symbols
            symbols = self.config.get_trading_symbols()
            historical_data = {}
            
            for symbol in symbols:
                data = self.data_processor.get_historical_data(
                    symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
                )
                if data is not None and not data.empty:
                    historical_data[symbol] = data
            
            if not historical_data:
                self.logger.error("No historical data available for backtesting")
                return self._create_empty_backtest_result()
            
            # Run simulation day by day
            backtest_state = self._simulate_trading(strategy, historical_data, backtest_state)
            
            # Calculate performance metrics
            results = self._calculate_performance_metrics(backtest_state, initial_capital)
            
            # Store results
            self.backtest_results[f"{strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"] = results
            
            self.logger.info(f"Backtest completed for {strategy}. Total return: {results.get('total_return', 0):.2%}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error running backtest for {strategy}: {str(e)}")
            return self._create_empty_backtest_result()
    
    def _simulate_trading(self, strategy: str, historical_data: Dict[str, pd.DataFrame], 
                         backtest_state: Dict) -> Dict:
        """Simulate trading based on strategy signals"""
        try:
            # Get all unique dates from historical data
            all_dates = set()
            for data in historical_data.values():
                all_dates.update(data.index.date)
            
            sorted_dates = sorted(all_dates)
            
            for current_date in sorted_dates:
                # Update portfolio values for current date
                self._update_portfolio_values(backtest_state, historical_data, current_date)
                
                # Generate signals for current date
                signals = self._generate_backtest_signals(strategy, historical_data, current_date)
                
                # Execute trades based on signals
                for signal in signals:
                    self._execute_backtest_trade(signal, backtest_state, current_date)
                
                # Record equity curve point
                total_value = self._calculate_total_portfolio_value(backtest_state, historical_data, current_date)
                backtest_state['equity_curve'].append({
                    'date': current_date,
                    'equity': total_value
                })
                
                # Calculate daily return
                if len(backtest_state['equity_curve']) > 1:
                    prev_value = backtest_state['equity_curve'][-2]['equity']
                    daily_return = (total_value - prev_value) / prev_value if prev_value > 0 else 0
                    backtest_state['daily_returns'].append(daily_return)
            
            return backtest_state
            
        except Exception as e:
            self.logger.error(f"Error in trading simulation: {str(e)}")
            return backtest_state
    
    def _generate_backtest_signals(self, strategy: str, historical_data: Dict[str, pd.DataFrame], 
                                  current_date) -> List[Dict]:
        """Generate trading signals for backtesting"""
        signals = []
        
        try:
            for symbol, data in historical_data.items():
                # Get data up to current date
                available_data = data[data.index.date <= current_date]
                
                if len(available_data) < 50:  # Need minimum data for signals
                    continue
                
                # Generate signals based on strategy type
                signal = self._generate_strategy_signal(strategy, symbol, available_data, current_date)
                if signal:
                    signals.append(signal)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating backtest signals: {str(e)}")
            return []
    
    def _generate_strategy_signal(self, strategy: str, symbol: str, data: pd.DataFrame, 
                                current_date) -> Optional[Dict]:
        """Generate signal based on strategy type"""
        try:
            latest = data.iloc[-1]
            
            if strategy == "Neural Network Momentum":
                return self._neural_momentum_signal(symbol, data, latest, current_date)
            elif strategy == "Ensemble Trend Following":
                return self._ensemble_trend_signal(symbol, data, latest, current_date)
            elif strategy == "AI Pattern Recognition":
                return self._ai_pattern_signal(symbol, data, latest, current_date)
            else:
                return self._simple_momentum_signal(symbol, data, latest, current_date)
                
        except Exception as e:
            self.logger.error(f"Error generating {strategy} signal for {symbol}: {str(e)}")
            return None
    
    def _neural_momentum_signal(self, symbol: str, data: pd.DataFrame, latest: pd.Series, 
                               current_date) -> Optional[Dict]:
        """Generate neural network momentum signal for backtesting"""
        try:
            # Calculate momentum indicators
            momentum_5 = data['Close'].pct_change(5).iloc[-1]
            momentum_20 = data['Close'].pct_change(20).iloc[-1]
            rsi = latest.get('RSI', 50)
            volume_ratio = latest.get('Volume_Ratio', 1)
            
            # Simple momentum strategy logic
            signal_strength = 0
            
            if momentum_5 > 0.02 and momentum_20 > 0 and rsi < 70 and volume_ratio > 1.2:
                signal_strength = min(0.9, 0.5 + momentum_5 * 5)
                action = 'BUY'
            elif momentum_5 < -0.02 and momentum_20 < 0 and rsi > 30:
                signal_strength = min(0.9, 0.5 + abs(momentum_5) * 5)
                action = 'SELL'
            else:
                return None
            
            return {
                'symbol': symbol,
                'action': action,
                'price': latest['Close'],
                'quantity': max(10, int(100 * signal_strength)),
                'confidence': signal_strength,
                'date': current_date,
                'strategy': 'neural_momentum'
            }
            
        except Exception as e:
            self.logger.error(f"Error in neural momentum signal: {str(e)}")
            return None
    
    def _ensemble_trend_signal(self, symbol: str, data: pd.DataFrame, latest: pd.Series, 
                              current_date) -> Optional[Dict]:
        """Generate ensemble trend following signal for backtesting"""
        try:
            sma_20 = latest.get('SMA_20', latest['Close'])
            sma_50 = latest.get('SMA_50', latest['Close'])
            macd = latest.get('MACD', 0)
            macd_signal = latest.get('MACD_Signal', 0)
            current_price = latest['Close']
            
            # Trend analysis
            trend_score = 0
            
            if current_price > sma_20 > sma_50:
                trend_score += 1
            elif current_price < sma_20 < sma_50:
                trend_score -= 1
            
            if macd > macd_signal and macd > 0:
                trend_score += 1
            elif macd < macd_signal and macd < 0:
                trend_score -= 1
            
            # Price momentum
            price_momentum = data['Close'].pct_change(10).iloc[-1]
            if price_momentum > 0.01:
                trend_score += 1
            elif price_momentum < -0.01:
                trend_score -= 1
            
            if trend_score >= 2:
                return {
                    'symbol': symbol,
                    'action': 'BUY',
                    'price': current_price,
                    'quantity': max(50, int(100 * (trend_score / 3))),
                    'confidence': min(0.9, 0.6 + trend_score * 0.1),
                    'date': current_date,
                    'strategy': 'ensemble_trend'
                }
            elif trend_score <= -2:
                return {
                    'symbol': symbol,
                    'action': 'SELL',
                    'price': current_price,
                    'quantity': max(50, int(100 * (abs(trend_score) / 3))),
                    'confidence': min(0.9, 0.6 + abs(trend_score) * 0.1),
                    'date': current_date,
                    'strategy': 'ensemble_trend'
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in ensemble trend signal: {str(e)}")
            return None
    
    def _ai_pattern_signal(self, symbol: str, data: pd.DataFrame, latest: pd.Series, 
                          current_date) -> Optional[Dict]:
        """Generate AI pattern recognition signal for backtesting"""
        try:
            # Simplified pattern recognition
            rsi = latest.get('RSI', 50)
            bb_upper = latest.get('BB_Upper', latest['Close'] * 1.02)
            bb_lower = latest.get('BB_Lower', latest['Close'] * 0.98)
            current_price = latest['Close']
            
            # Look for reversal patterns
            if current_price < bb_lower and rsi < 30:
                # Oversold reversal pattern
                return {
                    'symbol': symbol,
                    'action': 'BUY',
                    'price': current_price,
                    'quantity': 75,
                    'confidence': 0.7,
                    'date': current_date,
                    'strategy': 'ai_pattern'
                }
            elif current_price > bb_upper and rsi > 70:
                # Overbought reversal pattern
                return {
                    'symbol': symbol,
                    'action': 'SELL',
                    'price': current_price,
                    'quantity': 75,
                    'confidence': 0.7,
                    'date': current_date,
                    'strategy': 'ai_pattern'
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in AI pattern signal: {str(e)}")
            return None
    
    def _simple_momentum_signal(self, symbol: str, data: pd.DataFrame, latest: pd.Series, 
                               current_date) -> Optional[Dict]:
        """Generate simple momentum signal for backtesting"""
        try:
            # Simple moving average crossover
            sma_short = data['Close'].rolling(10).mean().iloc[-1]
            sma_long = data['Close'].rolling(30).mean().iloc[-1]
            current_price = latest['Close']
            
            if sma_short > sma_long * 1.01:  # 1% buffer
                return {
                    'symbol': symbol,
                    'action': 'BUY',
                    'price': current_price,
                    'quantity': 100,
                    'confidence': 0.6,
                    'date': current_date,
                    'strategy': 'simple_momentum'
                }
            elif sma_short < sma_long * 0.99:  # 1% buffer
                return {
                    'symbol': symbol,
                    'action': 'SELL',
                    'price': current_price,
                    'quantity': 100,
                    'confidence': 0.6,
                    'date': current_date,
                    'strategy': 'simple_momentum'
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in simple momentum signal: {str(e)}")
            return None
    
    def _execute_backtest_trade(self, signal: Dict, backtest_state: Dict, current_date):
        """Execute a trade in the backtest simulation"""
        try:
            symbol = signal['symbol']
            action = signal['action']
            price = signal['price']
            quantity = signal['quantity']
            
            # Calculate commission
            trade_value = abs(quantity) * price
            commission = max(1.0, trade_value * 0.001)  # 0.1% commission, minimum $1
            
            if action == 'BUY':
                # Check if we have enough capital
                total_cost = trade_value + commission
                if backtest_state['capital'] >= total_cost:
                    # Execute buy
                    backtest_state['capital'] -= total_cost
                    backtest_state['commission_paid'] += commission
                    
                    # Update position
                    if symbol in backtest_state['positions']:
                        # Average down/up
                        existing_qty = backtest_state['positions'][symbol]['quantity']
                        existing_price = backtest_state['positions'][symbol]['avg_price']
                        
                        new_qty = existing_qty + quantity
                        new_avg_price = ((existing_price * existing_qty) + (price * quantity)) / new_qty
                        
                        backtest_state['positions'][symbol] = {
                            'quantity': new_qty,
                            'avg_price': new_avg_price,
                            'current_price': price
                        }
                    else:
                        # New position
                        backtest_state['positions'][symbol] = {
                            'quantity': quantity,
                            'avg_price': price,
                            'current_price': price
                        }
                    
                    # Record trade
                    backtest_state['trades'].append({
                        'date': current_date,
                        'symbol': symbol,
                        'action': action,
                        'quantity': quantity,
                        'price': price,
                        'commission': commission,
                        'strategy': signal.get('strategy', 'unknown')
                    })
            
            elif action == 'SELL':
                # Check if we have the position
                if symbol in backtest_state['positions']:
                    position = backtest_state['positions'][symbol]
                    available_qty = position['quantity']
                    
                    # Sell what we can (up to available quantity)
                    sell_qty = min(quantity, available_qty)
                    
                    if sell_qty > 0:
                        # Execute sell
                        proceeds = sell_qty * price - commission
                        backtest_state['capital'] += proceeds
                        backtest_state['commission_paid'] += commission
                        
                        # Update position
                        remaining_qty = available_qty - sell_qty
                        if remaining_qty <= 0:
                            # Position closed
                            del backtest_state['positions'][symbol]
                        else:
                            # Partial close
                            backtest_state['positions'][symbol]['quantity'] = remaining_qty
                        
                        # Calculate realized P&L
                        cost_basis = position['avg_price'] * sell_qty
                        realized_pnl = proceeds - cost_basis
                        
                        # Record trade
                        backtest_state['trades'].append({
                            'date': current_date,
                            'symbol': symbol,
                            'action': action,
                            'quantity': sell_qty,
                            'price': price,
                            'commission': commission,
                            'realized_pnl': realized_pnl,
                            'strategy': signal.get('strategy', 'unknown')
                        })
            
        except Exception as e:
            self.logger.error(f"Error executing backtest trade: {str(e)}")
    
    def _update_portfolio_values(self, backtest_state: Dict, historical_data: Dict[str, pd.DataFrame], 
                               current_date):
        """Update current prices for all positions"""
        try:
            for symbol in backtest_state['positions']:
                if symbol in historical_data:
                    data = historical_data[symbol]
                    current_data = data[data.index.date <= current_date]
                    if not current_data.empty:
                        current_price = current_data.iloc[-1]['Close']
                        backtest_state['positions'][symbol]['current_price'] = current_price
        except Exception as e:
            self.logger.error(f"Error updating portfolio values: {str(e)}")
    
    def _calculate_total_portfolio_value(self, backtest_state: Dict, historical_data: Dict[str, pd.DataFrame], 
                                       current_date) -> float:
        """Calculate total portfolio value"""
        try:
            total_value = backtest_state['capital']
            
            for symbol, position in backtest_state['positions'].items():
                market_value = position['quantity'] * position['current_price']
                total_value += market_value
            
            return total_value
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio value: {str(e)}")
            return backtest_state['capital']
    
    def _calculate_performance_metrics(self, backtest_state: Dict, initial_capital: float) -> Dict:
        """Calculate comprehensive performance metrics"""
        try:
            equity_curve = pd.DataFrame(backtest_state['equity_curve'])
            trades_df = pd.DataFrame(backtest_state['trades'])
            
            if equity_curve.empty:
                return self._create_empty_backtest_result()
            
            final_value = equity_curve['equity'].iloc[-1]
            
            # Basic metrics
            total_return = (final_value - initial_capital) / initial_capital
            
            # Calculate returns
            equity_curve['returns'] = equity_curve['equity'].pct_change()
            returns = equity_curve['returns'].dropna()
            
            # Risk metrics
            if len(returns) > 1:
                volatility = returns.std() * np.sqrt(252)  # Annualized
                sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
                
                # Max drawdown
                cumulative = (1 + returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = cumulative / running_max - 1
                max_drawdown = drawdown.min()
            else:
                volatility = 0
                sharpe_ratio = 0
                max_drawdown = 0
            
            # Trade statistics
            if not trades_df.empty:
                total_trades = len(trades_df)
                winning_trades = len(trades_df[trades_df.get('realized_pnl', 0) > 0])
                win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
                
                # Average trade metrics
                avg_trade_pnl = trades_df.get('realized_pnl', [0]).mean()
                avg_winning_trade = trades_df[trades_df.get('realized_pnl', 0) > 0].get('realized_pnl', [0]).mean()
                avg_losing_trade = trades_df[trades_df.get('realized_pnl', 0) < 0].get('realized_pnl', [0]).mean()
                
                # Profit factor
                gross_profit = trades_df[trades_df.get('realized_pnl', 0) > 0].get('realized_pnl', [0]).sum()
                gross_loss = abs(trades_df[trades_df.get('realized_pnl', 0) < 0].get('realized_pnl', [0]).sum())
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            else:
                total_trades = 0
                win_rate = 0
                avg_trade_pnl = 0
                avg_winning_trade = 0
                avg_losing_trade = 0
                profit_factor = 0
            
            return {
                'total_return': total_return,
                'annualized_return': (1 + total_return) ** (252 / len(equity_curve)) - 1 if len(equity_curve) > 0 else 0,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': abs(max_drawdown),
                'final_value': final_value,
                'total_trades': total_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'avg_trade_pnl': avg_trade_pnl,
                'avg_winning_trade': avg_winning_trade,
                'avg_losing_trade': avg_losing_trade,
                'total_commission': backtest_state['commission_paid'],
                'equity_curve': equity_curve.to_dict('records'),
                'trades': trades_df.to_dict('records') if not trades_df.empty else []
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {str(e)}")
            return self._create_empty_backtest_result()
    
    def _create_empty_backtest_result(self) -> Dict:
        """Create empty backtest result structure"""
        return {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'final_value': 0.0,
            'total_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'avg_trade_pnl': 0.0,
            'avg_winning_trade': 0.0,
            'avg_losing_trade': 0.0,
            'total_commission': 0.0,
            'equity_curve': [],
            'trades': []
        }
    
    def get_backtest_results(self) -> Dict:
        """Get all backtest results"""
        return self.backtest_results
    
    def compare_strategies(self, strategy_results: List[str]) -> Dict:
        """Compare multiple strategy backtest results"""
        try:
            comparison = {}
            
            for strategy_key in strategy_results:
                if strategy_key in self.backtest_results:
                    result = self.backtest_results[strategy_key]
                    comparison[strategy_key] = {
                        'total_return': result.get('total_return', 0),
                        'sharpe_ratio': result.get('sharpe_ratio', 0),
                        'max_drawdown': result.get('max_drawdown', 0),
                        'win_rate': result.get('win_rate', 0),
                        'total_trades': result.get('total_trades', 0)
                    }
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Error comparing strategies: {str(e)}")
            return {}
