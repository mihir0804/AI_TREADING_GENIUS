import threading
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import queue

class TradeExecutor:
    """Handles trade execution with safety mechanisms and risk controls"""
    
    def __init__(self, config, logger, risk_manager, portfolio_manager):
        self.config = config
        self.logger = logger
        self.risk_manager = risk_manager
        self.portfolio_manager = portfolio_manager
        
        # Execution state
        self.is_trading_enabled = False
        self.emergency_stop_active = False
        self.paper_trading_mode = True
        
        # Trade queue and processing
        self.trade_queue = queue.Queue()
        self.execution_thread = None
        self.stop_loss_orders = {}  # symbol -> stop loss info
        self.take_profit_orders = {}  # symbol -> take profit info
        
        # Execution statistics
        self.total_trades_executed = 0
        self.successful_trades = 0
        self.failed_trades = 0
        self.total_commission_paid = 0.0
        
        # Safety mechanisms
        self.max_trades_per_minute = 10
        self.trade_count_window = []
        
    def start_trading(self, paper_mode: bool = True):
        """Start the trade execution engine"""
        try:
            self.paper_trading_mode = paper_mode
            self.is_trading_enabled = True
            self.emergency_stop_active = False
            
            # Start execution thread
            if self.execution_thread is None or not self.execution_thread.is_alive():
                self.execution_thread = threading.Thread(target=self._execution_loop, daemon=True)
                self.execution_thread.start()
            
            mode = "Paper Trading" if paper_mode else "Live Trading"
            self.logger.info(f"Trade execution started in {mode} mode")
            
        except Exception as e:
            self.logger.error(f"Error starting trade execution: {str(e)}")
            self.is_trading_enabled = False
    
    def stop_trading(self):
        """Stop the trade execution engine"""
        try:
            self.is_trading_enabled = False
            self.logger.info("Trade execution stopped")
        except Exception as e:
            self.logger.error(f"Error stopping trade execution: {str(e)}")
    
    def emergency_stop(self):
        """Activate emergency stop - immediately halt all trading"""
        try:
            self.emergency_stop_active = True
            self.is_trading_enabled = False
            
            # Cancel all pending orders
            self._cancel_all_pending_orders()
            
            # Close all positions if in live trading mode
            if not self.paper_trading_mode:
                self._emergency_close_all_positions()
            
            self.logger.critical("EMERGENCY STOP ACTIVATED - All trading halted")
            
        except Exception as e:
            self.logger.error(f"Error during emergency stop: {str(e)}")
    
    def execute_trades(self, signals: List) -> List[Dict]:
        """Queue trades for execution"""
        execution_results = []
        
        try:
            if not self.is_trading_enabled or self.emergency_stop_active:
                self.logger.warning("Trading is disabled - signals not executed")
                return execution_results
            
            # Rate limiting check
            if not self._check_rate_limits():
                self.logger.warning("Rate limit exceeded - signals queued for later")
                return execution_results
            
            for signal in signals:
                try:
                    # Validate signal with risk manager
                    is_valid, reason, adjusted_params = self.risk_manager.validate_trade(
                        signal, self.portfolio_manager
                    )
                    
                    if not is_valid:
                        self.logger.warning(f"Trade rejected: {reason}")
                        execution_results.append({
                            'signal': signal,
                            'status': 'rejected',
                            'reason': reason,
                            'timestamp': datetime.now()
                        })
                        continue
                    
                    # Queue validated trade for execution
                    trade_order = {
                        'signal': signal,
                        'adjusted_params': adjusted_params,
                        'timestamp': datetime.now(),
                        'retry_count': 0
                    }
                    
                    self.trade_queue.put(trade_order)
                    
                    execution_results.append({
                        'signal': signal,
                        'status': 'queued',
                        'reason': 'Trade queued for execution',
                        'timestamp': datetime.now()
                    })
                    
                except Exception as e:
                    self.logger.error(f"Error processing signal for {signal.symbol}: {str(e)}")
                    execution_results.append({
                        'signal': signal,
                        'status': 'error',
                        'reason': str(e),
                        'timestamp': datetime.now()
                    })
            
            return execution_results
            
        except Exception as e:
            self.logger.error(f"Error executing trades: {str(e)}")
            return execution_results
    
    def _execution_loop(self):
        """Main execution loop running in separate thread"""
        try:
            while self.is_trading_enabled and not self.emergency_stop_active:
                try:
                    # Process trade queue
                    if not self.trade_queue.empty():
                        trade_order = self.trade_queue.get(timeout=1)
                        self._execute_single_trade(trade_order)
                    
                    # Check stop loss and take profit orders
                    self._check_stop_loss_orders()
                    self._check_take_profit_orders()
                    
                    # Emergency stop check
                    if self.risk_manager.emergency_stop_check(self.portfolio_manager):
                        self.emergency_stop()
                        break
                    
                    time.sleep(0.1)  # Small delay to prevent excessive CPU usage
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    self.logger.error(f"Error in execution loop: {str(e)}")
                    time.sleep(1)
            
        except Exception as e:
            self.logger.error(f"Critical error in execution loop: {str(e)}")
        finally:
            self.logger.info("Execution loop stopped")
    
    def _execute_single_trade(self, trade_order: Dict):
        """Execute a single trade order"""
        try:
            signal = trade_order['signal']
            adjusted_params = trade_order.get('adjusted_params', {})
            
            # Use adjusted quantity if available
            quantity = adjusted_params.get('adjusted_quantity', signal.quantity)
            
            if self.paper_trading_mode:
                # Paper trading execution
                success = self._execute_paper_trade(signal, quantity, adjusted_params)
            else:
                # Live trading execution (would integrate with broker API)
                success = self._execute_live_trade(signal, quantity, adjusted_params)
            
            # Update statistics
            self.total_trades_executed += 1
            if success:
                self.successful_trades += 1
                
                # Set up stop loss and take profit orders
                self._setup_risk_orders(signal, adjusted_params)
                
            else:
                self.failed_trades += 1
                
                # Retry logic for failed trades
                if trade_order['retry_count'] < 3:
                    trade_order['retry_count'] += 1
                    self.trade_queue.put(trade_order)
                    self.logger.info(f"Retrying trade for {signal.symbol} (attempt {trade_order['retry_count']})")
            
        except Exception as e:
            self.logger.error(f"Error executing trade for {signal.symbol}: {str(e)}")
    
    def _execute_paper_trade(self, signal, quantity: int, adjusted_params: Dict) -> bool:
        """Execute trade in paper trading mode"""
        try:
            # Simulate trade execution
            success = self.portfolio_manager.add_position(
                symbol=signal.symbol,
                quantity=quantity if signal.action == 'BUY' else -quantity,
                price=signal.price,
                position_type='long' if signal.action == 'BUY' else 'short'
            )
            
            if success:
                self.logger.info(f"Paper trade executed: {signal.action} {quantity} {signal.symbol} @ ${signal.price:.2f}")
                
                # Update trade statistics for strategy performance tracking
                self._update_strategy_performance(signal, success=True)
                
            return success
            
        except Exception as e:
            self.logger.error(f"Error in paper trade execution: {str(e)}")
            return False
    
    def _execute_live_trade(self, signal, quantity: int, adjusted_params: Dict) -> bool:
        """Execute trade in live trading mode"""
        try:
            # In a real implementation, this would integrate with broker APIs
            # For now, we'll simulate the execution but log it as live trading
            
            self.logger.warning("Live trading not implemented - using paper trading simulation")
            
            # Simulate broker API call
            success = self._simulate_broker_execution(signal, quantity)
            
            if success:
                # Update portfolio with executed trade
                portfolio_success = self.portfolio_manager.add_position(
                    symbol=signal.symbol,
                    quantity=quantity if signal.action == 'BUY' else -quantity,
                    price=signal.price,
                    position_type='long' if signal.action == 'BUY' else 'short'
                )
                
                if portfolio_success:
                    self.logger.info(f"Live trade executed: {signal.action} {quantity} {signal.symbol} @ ${signal.price:.2f}")
                    self._update_strategy_performance(signal, success=True)
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error in live trade execution: {str(e)}")
            return False
    
    def _simulate_broker_execution(self, signal, quantity: int) -> bool:
        """Simulate broker API execution (placeholder for real broker integration)"""
        try:
            # Simulate execution delay
            time.sleep(0.1)
            
            # Simulate 99% success rate
            import random
            success_rate = 0.99
            
            if random.random() < success_rate:
                # Calculate commission
                trade_value = quantity * signal.price
                commission = max(1.0, trade_value * 0.001)
                self.total_commission_paid += commission
                
                return True
            else:
                self.logger.warning(f"Simulated broker execution failed for {signal.symbol}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error in broker simulation: {str(e)}")
            return False
    
    def _setup_risk_orders(self, signal, adjusted_params: Dict):
        """Set up stop loss and take profit orders"""
        try:
            symbol = signal.symbol
            
            # Stop loss order
            stop_loss_price = adjusted_params.get('stop_loss_price')
            if stop_loss_price:
                self.stop_loss_orders[symbol] = {
                    'price': stop_loss_price,
                    'action': 'SELL' if signal.action == 'BUY' else 'BUY',
                    'created_at': datetime.now(),
                    'original_signal': signal
                }
            
            # Take profit order
            take_profit_price = adjusted_params.get('take_profit_price')
            if take_profit_price:
                self.take_profit_orders[symbol] = {
                    'price': take_profit_price,
                    'action': 'SELL' if signal.action == 'BUY' else 'BUY',
                    'created_at': datetime.now(),
                    'original_signal': signal
                }
            
        except Exception as e:
            self.logger.error(f"Error setting up risk orders: {str(e)}")
    
    def _check_stop_loss_orders(self):
        """Check and execute stop loss orders"""
        try:
            from trading_system.data_processor import DataProcessor
            data_processor = DataProcessor(self.config, self.logger)
            
            orders_to_remove = []
            
            for symbol, order in self.stop_loss_orders.items():
                try:
                    # Get current price
                    data = data_processor.get_market_data(symbol, period="1d", interval="1m")
                    if data is None or data.empty:
                        continue
                    
                    current_price = data['Close'].iloc[-1]
                    
                    # Check if stop loss should trigger
                    should_trigger = False
                    
                    if order['action'] == 'SELL' and current_price <= order['price']:
                        should_trigger = True
                    elif order['action'] == 'BUY' and current_price >= order['price']:
                        should_trigger = True
                    
                    if should_trigger:
                        # Execute stop loss
                        position = self.portfolio_manager.get_position(symbol)
                        if position:
                            success = self.portfolio_manager.close_position(symbol, current_price)
                            if success:
                                self.logger.info(f"Stop loss executed for {symbol} @ ${current_price:.2f}")
                                orders_to_remove.append(symbol)
                            
                except Exception as e:
                    self.logger.error(f"Error checking stop loss for {symbol}: {str(e)}")
                    continue
            
            # Remove executed orders
            for symbol in orders_to_remove:
                del self.stop_loss_orders[symbol]
                if symbol in self.take_profit_orders:
                    del self.take_profit_orders[symbol]
                    
        except Exception as e:
            self.logger.error(f"Error checking stop loss orders: {str(e)}")
    
    def _check_take_profit_orders(self):
        """Check and execute take profit orders"""
        try:
            from trading_system.data_processor import DataProcessor
            data_processor = DataProcessor(self.config, self.logger)
            
            orders_to_remove = []
            
            for symbol, order in self.take_profit_orders.items():
                try:
                    # Get current price
                    data = data_processor.get_market_data(symbol, period="1d", interval="1m")
                    if data is None or data.empty:
                        continue
                    
                    current_price = data['Close'].iloc[-1]
                    
                    # Check if take profit should trigger
                    should_trigger = False
                    
                    if order['action'] == 'SELL' and current_price >= order['price']:
                        should_trigger = True
                    elif order['action'] == 'BUY' and current_price <= order['price']:
                        should_trigger = True
                    
                    if should_trigger:
                        # Execute take profit
                        position = self.portfolio_manager.get_position(symbol)
                        if position:
                            success = self.portfolio_manager.close_position(symbol, current_price)
                            if success:
                                self.logger.info(f"Take profit executed for {symbol} @ ${current_price:.2f}")
                                orders_to_remove.append(symbol)
                            
                except Exception as e:
                    self.logger.error(f"Error checking take profit for {symbol}: {str(e)}")
                    continue
            
            # Remove executed orders
            for symbol in orders_to_remove:
                del self.take_profit_orders[symbol]
                if symbol in self.stop_loss_orders:
                    del self.stop_loss_orders[symbol]
                    
        except Exception as e:
            self.logger.error(f"Error checking take profit orders: {str(e)}")
    
    def _check_rate_limits(self) -> bool:
        """Check if we're within rate limits for trading"""
        try:
            current_time = datetime.now()
            
            # Remove old entries (older than 1 minute)
            self.trade_count_window = [
                t for t in self.trade_count_window 
                if (current_time - t).seconds < 60
            ]
            
            # Check if we can add another trade
            if len(self.trade_count_window) >= self.max_trades_per_minute:
                return False
            
            # Add current time to window
            self.trade_count_window.append(current_time)
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking rate limits: {str(e)}")
            return False
    
    def _cancel_all_pending_orders(self):
        """Cancel all pending stop loss and take profit orders"""
        try:
            self.stop_loss_orders.clear()
            self.take_profit_orders.clear()
            
            # Clear trade queue
            while not self.trade_queue.empty():
                try:
                    self.trade_queue.get_nowait()
                except queue.Empty:
                    break
            
            self.logger.info("All pending orders cancelled")
            
        except Exception as e:
            self.logger.error(f"Error cancelling pending orders: {str(e)}")
    
    def _emergency_close_all_positions(self):
        """Emergency close all open positions"""
        try:
            positions = self.portfolio_manager.get_all_positions()
            
            for symbol in positions.keys():
                try:
                    success = self.portfolio_manager.close_position(symbol)
                    if success:
                        self.logger.info(f"Emergency closed position in {symbol}")
                    else:
                        self.logger.error(f"Failed to emergency close position in {symbol}")
                except Exception as e:
                    self.logger.error(f"Error emergency closing {symbol}: {str(e)}")
            
        except Exception as e:
            self.logger.error(f"Error during emergency close: {str(e)}")
    
    def _update_strategy_performance(self, signal, success: bool):
        """Update strategy performance tracking"""
        try:
            # This would integrate with the strategy manager to update performance metrics
            # For now, just log the trade
            status = "successful" if success else "failed"
            self.logger.info(f"Strategy {signal.strategy} trade {status}: {signal.action} {signal.symbol}")
            
        except Exception as e:
            self.logger.error(f"Error updating strategy performance: {str(e)}")
    
    def get_execution_statistics(self) -> Dict:
        """Get trade execution statistics"""
        try:
            success_rate = (self.successful_trades / max(1, self.total_trades_executed)) * 100
            
            return {
                'total_trades_executed': self.total_trades_executed,
                'successful_trades': self.successful_trades,
                'failed_trades': self.failed_trades,
                'success_rate': success_rate,
                'total_commission_paid': self.total_commission_paid,
                'pending_stop_losses': len(self.stop_loss_orders),
                'pending_take_profits': len(self.take_profit_orders),
                'trades_in_queue': self.trade_queue.qsize(),
                'trading_enabled': self.is_trading_enabled,
                'paper_trading_mode': self.paper_trading_mode,
                'emergency_stop_active': self.emergency_stop_active
            }
            
        except Exception as e:
            self.logger.error(f"Error getting execution statistics: {str(e)}")
            return {}
    
    def get_pending_orders(self) -> Dict:
        """Get all pending stop loss and take profit orders"""
        try:
            return {
                'stop_loss_orders': dict(self.stop_loss_orders),
                'take_profit_orders': dict(self.take_profit_orders)
            }
        except Exception as e:
            self.logger.error(f"Error getting pending orders: {str(e)}")
            return {'stop_loss_orders': {}, 'take_profit_orders': {}}
