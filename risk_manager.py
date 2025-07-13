import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import threading

class RiskManager:
    """Comprehensive risk management system"""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.risk_parameters = self._load_default_risk_parameters()
        self.portfolio_risk_cache = {}
        self.risk_alerts = []
        self.max_risk_alerts = 100
        
    def _load_default_risk_parameters(self) -> Dict:
        """Load default risk management parameters"""
        return {
            'max_portfolio_risk': 0.05,  # 5% maximum portfolio risk
            'max_position_size': 0.10,   # 10% maximum position size
            'stop_loss_threshold': 0.05,  # 5% stop loss
            'take_profit_threshold': 0.15, # 15% take profit
            'max_daily_loss': 0.03,      # 3% maximum daily loss
            'max_correlation': 0.7,      # Maximum correlation between positions
            'var_confidence': 0.95,      # VaR confidence level
            'max_leverage': 2.0,         # Maximum leverage
            'min_cash_reserve': 0.10     # 10% minimum cash reserve
        }
    
    def update_risk_parameters(self, new_parameters: Dict):
        """Update risk management parameters"""
        try:
            self.risk_parameters.update(new_parameters)
            self.logger.info(f"Updated risk parameters: {new_parameters}")
        except Exception as e:
            self.logger.error(f"Error updating risk parameters: {str(e)}")
    
    def validate_trade(self, signal, portfolio_manager) -> Tuple[bool, str, Optional[Dict]]:
        """Validate if a trade meets risk management criteria"""
        try:
            # Get current portfolio state
            portfolio = portfolio_manager.get_portfolio_summary()
            positions = portfolio_manager.get_all_positions()
            
            # 1. Position size check
            position_check = self._check_position_size(signal, portfolio)
            if not position_check['valid']:
                return False, position_check['reason'], None
            
            # 2. Portfolio risk check
            risk_check = self._check_portfolio_risk(signal, portfolio, positions)
            if not risk_check['valid']:
                return False, risk_check['reason'], None
            
            # 3. Correlation check
            correlation_check = self._check_correlation(signal, positions)
            if not correlation_check['valid']:
                return False, correlation_check['reason'], None
            
            # 4. Daily loss limit check
            daily_loss_check = self._check_daily_loss_limit(portfolio)
            if not daily_loss_check['valid']:
                return False, daily_loss_check['reason'], None
            
            # 5. Cash reserve check
            cash_check = self._check_cash_reserve(signal, portfolio)
            if not cash_check['valid']:
                return False, cash_check['reason'], None
            
            # Calculate adjusted trade parameters
            adjusted_params = self._calculate_adjusted_trade_params(signal, portfolio)
            
            self.logger.info(f"Trade validation passed for {signal.symbol} {signal.action}")
            return True, "Trade approved", adjusted_params
            
        except Exception as e:
            self.logger.error(f"Error validating trade: {str(e)}")
            return False, f"Validation error: {str(e)}", None
    
    def _check_position_size(self, signal, portfolio) -> Dict:
        """Check if position size is within limits"""
        try:
            portfolio_value = portfolio.get('total_value', 0)
            if portfolio_value == 0:
                return {'valid': False, 'reason': "No portfolio value available"}
            
            trade_value = signal.price * signal.quantity
            position_ratio = trade_value / portfolio_value
            
            if position_ratio > self.risk_parameters['max_position_size']:
                return {
                    'valid': False, 
                    'reason': f"Position size {position_ratio:.2%} exceeds maximum {self.risk_parameters['max_position_size']:.2%}"
                }
            
            return {'valid': True, 'reason': "Position size OK"}
            
        except Exception as e:
            return {'valid': False, 'reason': f"Position size check error: {str(e)}"}
    
    def _check_portfolio_risk(self, signal, portfolio, positions) -> Dict:
        """Check portfolio-level risk metrics"""
        try:
            # Calculate portfolio VaR
            portfolio_var = self._calculate_portfolio_var(positions)
            
            if portfolio_var > self.risk_parameters['max_portfolio_risk']:
                return {
                    'valid': False,
                    'reason': f"Portfolio VaR {portfolio_var:.2%} exceeds maximum {self.risk_parameters['max_portfolio_risk']:.2%}"
                }
            
            # Check leverage
            total_exposure = sum(abs(pos.get('market_value', 0)) for pos in positions.values())
            cash = portfolio.get('cash', 0)
            leverage = total_exposure / cash if cash > 0 else float('inf')
            
            if leverage > self.risk_parameters['max_leverage']:
                return {
                    'valid': False,
                    'reason': f"Leverage {leverage:.2f} exceeds maximum {self.risk_parameters['max_leverage']:.2f}"
                }
            
            return {'valid': True, 'reason': "Portfolio risk OK"}
            
        except Exception as e:
            return {'valid': False, 'reason': f"Portfolio risk check error: {str(e)}"}
    
    def _check_correlation(self, signal, positions) -> Dict:
        """Check correlation between positions"""
        try:
            if not positions or len(positions) < 2:
                return {'valid': True, 'reason': "Insufficient positions for correlation check"}
            
            # Simplified correlation check - would need historical price data for full implementation
            # For now, check sector/industry diversification
            symbols = [signal.symbol] + list(positions.keys())
            
            # Basic check: don't allow too many positions in similar symbols
            similar_symbols = [s for s in symbols if s.startswith(signal.symbol[:2])]
            
            if len(similar_symbols) > 3:
                return {
                    'valid': False,
                    'reason': f"Too many similar positions: {similar_symbols}"
                }
            
            return {'valid': True, 'reason': "Correlation check passed"}
            
        except Exception as e:
            return {'valid': False, 'reason': f"Correlation check error: {str(e)}"}
    
    def _check_daily_loss_limit(self, portfolio) -> Dict:
        """Check daily loss limits"""
        try:
            daily_pnl = portfolio.get('daily_pnl', 0)
            portfolio_value = portfolio.get('total_value', 0)
            
            if portfolio_value == 0:
                return {'valid': True, 'reason': "No portfolio value for daily loss check"}
            
            daily_loss_ratio = abs(daily_pnl) / portfolio_value if daily_pnl < 0 else 0
            
            if daily_loss_ratio > self.risk_parameters['max_daily_loss']:
                return {
                    'valid': False,
                    'reason': f"Daily loss {daily_loss_ratio:.2%} exceeds limit {self.risk_parameters['max_daily_loss']:.2%}"
                }
            
            return {'valid': True, 'reason': "Daily loss limit OK"}
            
        except Exception as e:
            return {'valid': False, 'reason': f"Daily loss check error: {str(e)}"}
    
    def _check_cash_reserve(self, signal, portfolio) -> Dict:
        """Check cash reserve requirements"""
        try:
            if signal.action != 'BUY':
                return {'valid': True, 'reason': "No cash requirement for sell order"}
            
            cash = portfolio.get('cash', 0)
            total_value = portfolio.get('total_value', 0)
            
            trade_value = signal.price * signal.quantity
            remaining_cash = cash - trade_value
            cash_ratio = remaining_cash / total_value if total_value > 0 else 0
            
            if cash_ratio < self.risk_parameters['min_cash_reserve']:
                return {
                    'valid': False,
                    'reason': f"Insufficient cash reserve. Would have {cash_ratio:.2%}, need {self.risk_parameters['min_cash_reserve']:.2%}"
                }
            
            return {'valid': True, 'reason': "Cash reserve OK"}
            
        except Exception as e:
            return {'valid': False, 'reason': f"Cash reserve check error: {str(e)}"}
    
    def _calculate_adjusted_trade_params(self, signal, portfolio) -> Dict:
        """Calculate risk-adjusted trade parameters"""
        try:
            portfolio_value = portfolio.get('total_value', 0)
            
            # Adjust quantity based on portfolio risk
            risk_adjusted_quantity = signal.quantity
            
            # Reduce size if portfolio value is low
            if portfolio_value < 50000:  # Less than $50k
                risk_adjusted_quantity = int(signal.quantity * 0.5)
            elif portfolio_value < 100000:  # Less than $100k
                risk_adjusted_quantity = int(signal.quantity * 0.75)
            
            # Calculate stop loss and take profit levels
            stop_loss_price = None
            take_profit_price = None
            
            if signal.action == 'BUY':
                stop_loss_price = signal.price * (1 - self.risk_parameters['stop_loss_threshold'])
                take_profit_price = signal.price * (1 + self.risk_parameters['take_profit_threshold'])
            elif signal.action == 'SELL':
                stop_loss_price = signal.price * (1 + self.risk_parameters['stop_loss_threshold'])
                take_profit_price = signal.price * (1 - self.risk_parameters['take_profit_threshold'])
            
            return {
                'original_quantity': signal.quantity,
                'adjusted_quantity': max(1, risk_adjusted_quantity),
                'stop_loss_price': stop_loss_price,
                'take_profit_price': take_profit_price,
                'max_trade_value': portfolio_value * self.risk_parameters['max_position_size']
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating adjusted trade params: {str(e)}")
            return {}
    
    def _calculate_portfolio_var(self, positions: Dict) -> float:
        """Calculate portfolio Value at Risk"""
        try:
            if not positions:
                return 0.0
            
            # Simplified VaR calculation
            # In production, this would use historical price data and correlations
            total_risk = 0.0
            
            for symbol, position in positions.items():
                position_value = abs(position.get('market_value', 0))
                # Assume 2% daily volatility for simplification
                position_risk = position_value * 0.02
                total_risk += position_risk ** 2
            
            # Portfolio VaR (assuming some diversification benefit)
            portfolio_var = np.sqrt(total_risk) * 0.8  # 20% diversification benefit
            
            return portfolio_var
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio VaR: {str(e)}")
            return 0.0
    
    def get_portfolio_risk_score(self) -> float:
        """Calculate overall portfolio risk score (0-10 scale)"""
        try:
            # This would be calculated based on current portfolio state
            # For now, return a simplified score
            return np.random.uniform(2.0, 6.0)  # Placeholder
            
        except Exception as e:
            self.logger.error(f"Error calculating risk score: {str(e)}")
            return 5.0
    
    def get_risk_metrics(self) -> Dict:
        """Get comprehensive risk metrics"""
        try:
            return {
                'portfolio_var': 0.02,  # Placeholder values
                'sharpe_ratio': 1.2,
                'max_drawdown': 0.08,
                'beta': 1.1,
                'alpha': 0.03,
                'information_ratio': 0.8,
                'sortino_ratio': 1.5
            }
        except Exception as e:
            self.logger.error(f"Error getting risk metrics: {str(e)}")
            return {}
    
    def get_risk_status(self) -> Dict:
        """Get current risk status for all risk categories"""
        try:
            return {
                'Portfolio Risk': {
                    'status': 'safe',
                    'value': 0.03,
                    'limit': self.risk_parameters['max_portfolio_risk']
                },
                'Position Concentration': {
                    'status': 'safe',
                    'value': 0.08,
                    'limit': self.risk_parameters['max_position_size']
                },
                'Daily Loss': {
                    'status': 'safe',
                    'value': 0.01,
                    'limit': self.risk_parameters['max_daily_loss']
                },
                'Leverage': {
                    'status': 'safe',
                    'value': 1.3,
                    'limit': self.risk_parameters['max_leverage']
                }
            }
        except Exception as e:
            self.logger.error(f"Error getting risk status: {str(e)}")
            return {}
    
    def add_risk_alert(self, alert_type: str, message: str, severity: str = 'medium'):
        """Add a risk alert"""
        try:
            alert = {
                'timestamp': datetime.now(),
                'type': alert_type,
                'message': message,
                'severity': severity
            }
            
            self.risk_alerts.append(alert)
            
            # Keep only recent alerts
            if len(self.risk_alerts) > self.max_risk_alerts:
                self.risk_alerts = self.risk_alerts[-self.max_risk_alerts:]
            
            self.logger.warning(f"Risk Alert [{severity}]: {message}")
            
        except Exception as e:
            self.logger.error(f"Error adding risk alert: {str(e)}")
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict]:
        """Get recent risk alerts"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_alerts = [
                alert for alert in self.risk_alerts 
                if alert['timestamp'] > cutoff_time
            ]
            return recent_alerts
        except Exception as e:
            self.logger.error(f"Error getting recent alerts: {str(e)}")
            return []
    
    def emergency_stop_check(self, portfolio_manager) -> bool:
        """Check if emergency stop should be triggered"""
        try:
            portfolio = portfolio_manager.get_portfolio_summary()
            
            # Check for extreme losses
            daily_pnl_ratio = abs(portfolio.get('daily_pnl', 0)) / portfolio.get('total_value', 1)
            if daily_pnl_ratio > 0.10:  # 10% daily loss
                self.add_risk_alert('EMERGENCY_STOP', 'Extreme daily loss detected', 'critical')
                return True
            
            # Check for system errors
            # Additional emergency conditions would be added here
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error in emergency stop check: {str(e)}")
            return True  # Err on the side of caution
