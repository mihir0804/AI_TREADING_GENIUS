import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

class TradingConfig:
    """Centralized configuration management for the trading system"""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self._config = self._load_config()
        self._default_config = self._get_default_config()
        
        # Ensure all required keys exist
        self._validate_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values"""
        return {
            # Trading Parameters (All amounts in INR)
            "trading": {
                "initial_capital": 8300000.0,  # ₹83 Lakhs (equivalent to $100k USD)
                "max_position_size": 0.10,  # 10% of portfolio
                "commission_rate": 0.001,   # 0.1%
                "min_commission": 83.0,     # ₹83 (equivalent to $1 USD)
                "max_trades_per_day": 50,
                "min_signal_confidence": 0.6,
                "max_signals_per_cycle": 10
            },
            
            # Risk Management
            "risk_management": {
                "max_portfolio_risk": 0.05,     # 5%
                "max_daily_loss": 0.03,         # 3%
                "stop_loss_threshold": 0.05,    # 5%
                "take_profit_threshold": 0.15,  # 15%
                "max_correlation": 0.7,
                "var_confidence": 0.95,
                "max_leverage": 2.0,
                "min_cash_reserve": 0.10,       # 10%
                "emergency_stop_loss": 0.15     # 15%
            },
            
            # Indian Stock Symbols (NSE/BSE via Yahoo Finance)
            "symbols": {
                "primary": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS"],
                "secondary": ["ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "ASIANPAINT.NS", "MARUTI.NS"],
                "sectors": ["NIFTY50.NS", "BANKNIFTY.NS", "CNXIT.NS", "NIFTYFMCG.NS", "NIFTYAUTO.NS"],
                "crypto": [],  # Add crypto symbols if needed
                "forex": ["USDINR=X"]    # USD/INR exchange rate
            },
            
            # Machine Learning
            "ml_models": {
                "retrain_frequency_hours": 24,
                "min_training_samples": 1000,
                "validation_split": 0.2,
                "max_features": 20,
                "ensemble_weights": {
                    "random_forest": 0.3,
                    "gradient_boosting": 0.3,
                    "neural_network": 0.2,
                    "lstm": 0.2
                }
            },
            
            # Data Processing
            "data": {
                "update_frequency_minutes": 1,
                "historical_lookback_days": 252,  # 1 year
                "cache_expiry_minutes": 5,
                "max_api_calls_per_minute": 100,
                "data_sources": {
                    "primary": "yahoo_finance",
                    "backup": "alpha_vantage"
                }
            },
            
            # AI Advisor
            "ai_advisor": {
                "analysis_frequency_minutes": 15,
                "max_insights_per_session": 5,
                "confidence_threshold": 0.7,
                "cache_expiry_minutes": 5
            },
            
            # Backtesting
            "backtesting": {
                "default_start_date": "2023-01-01",
                "default_end_date": "2024-12-31",
                "default_initial_capital": 100000.0,
                "commission_rate": 0.001,
                "slippage": 0.001,  # 0.1%
                "benchmark": "SPY"
            },
            
            # Logging
            "logging": {
                "log_level": "INFO",
                "max_file_size_mb": 10,
                "backup_count": 5,
                "log_retention_days": 30
            },
            
            # System
            "system": {
                "max_threads": 4,
                "timeout_seconds": 30,
                "retry_attempts": 3,
                "health_check_interval_minutes": 5
            },
            
            # External APIs
            "api_keys": {
                "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
                "alpha_vantage_key": os.getenv("ALPHA_VANTAGE_API_KEY", ""),
                "polygon_key": os.getenv("POLYGON_API_KEY", ""),
                "broker_api_key": os.getenv("BROKER_API_KEY", "")
            },
            
            # Strategy Configuration
            "strategies": {
                "neural_momentum": {
                    "enabled": True,
                    "weight": 0.25,
                    "lookback_period": 20,
                    "min_momentum": 0.02
                },
                "ensemble_trend": {
                    "enabled": True,
                    "weight": 0.25,
                    "trend_threshold": 2,
                    "confirmation_period": 5
                },
                "ai_pattern": {
                    "enabled": True,
                    "weight": 0.20,
                    "pattern_confidence": 0.7
                },
                "adaptive_mean_reversion": {
                    "enabled": True,
                    "weight": 0.15,
                    "z_score_threshold": 1.5,
                    "lookback_period": 20
                },
                "multi_factor": {
                    "enabled": True,
                    "weight": 0.15,
                    "factor_weights": {
                        "momentum": 0.3,
                        "value": 0.25,
                        "quality": 0.25,
                        "volatility": 0.2
                    }
                }
            }
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            else:
                # Create default config file
                default_config = self._get_default_config()
                self._save_config(default_config)
                return default_config
        except Exception as e:
            print(f"Error loading config: {e}. Using default configuration.")
            return self._get_default_config()
    
    def _save_config(self, config: Dict[str, Any]):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=4, default=str)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def _validate_config(self):
        """Validate configuration and add missing keys"""
        def merge_configs(default: Dict, current: Dict) -> Dict:
            """Recursively merge default config with current config"""
            for key, value in default.items():
                if key not in current:
                    current[key] = value
                elif isinstance(value, dict) and isinstance(current[key], dict):
                    current[key] = merge_configs(value, current[key])
            return current
        
        self._config = merge_configs(self._default_config, self._config)
        self._save_config(self._config)
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'trading.initial_capital')"""
        try:
            keys = key_path.split('.')
            value = self._config
            
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return default
            
            return value
        except Exception as e:
            print(f"Error getting config value for {key_path}: {e}")
            return default
    
    def set(self, key_path: str, value: Any) -> bool:
        """Set configuration value using dot notation"""
        try:
            keys = key_path.split('.')
            config = self._config
            
            # Navigate to the parent of the target key
            for key in keys[:-1]:
                if key not in config:
                    config[key] = {}
                config = config[key]
            
            # Set the value
            config[keys[-1]] = value
            
            # Save to file
            self._save_config(self._config)
            return True
            
        except Exception as e:
            print(f"Error setting config value for {key_path}: {e}")
            return False
    
    def get_trading_symbols(self) -> List[str]:
        """Get all trading symbols"""
        symbols = []
        symbols.extend(self.get('symbols.primary', []))
        symbols.extend(self.get('symbols.secondary', []))
        symbols.extend(self.get('symbols.sectors', []))
        return list(set(symbols))  # Remove duplicates
    
    def get_strategy_config(self, strategy_name: str) -> Dict[str, Any]:
        """Get configuration for a specific strategy"""
        return self.get(f'strategies.{strategy_name}', {})
    
    def is_strategy_enabled(self, strategy_name: str) -> bool:
        """Check if a strategy is enabled"""
        return self.get(f'strategies.{strategy_name}.enabled', False)
    
    def get_risk_parameters(self) -> Dict[str, Any]:
        """Get risk management parameters"""
        return self.get('risk_management', {})
    
    def get_ml_config(self) -> Dict[str, Any]:
        """Get machine learning configuration"""
        return self.get('ml_models', {})
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data processing configuration"""
        return self.get('data', {})
    
    def get_api_key(self, service: str) -> str:
        """Get API key for a service"""
        return self.get(f'api_keys.{service}_api_key', '')
    
    def update_strategy_weights(self, weights: Dict[str, float]) -> bool:
        """Update strategy weights"""
        try:
            for strategy, weight in weights.items():
                self.set(f'strategies.{strategy}.weight', weight)
            return True
        except Exception as e:
            print(f"Error updating strategy weights: {e}")
            return False
    
    def update_risk_parameters(self, risk_params: Dict[str, Any]) -> bool:
        """Update risk management parameters"""
        try:
            for param, value in risk_params.items():
                self.set(f'risk_management.{param}', value)
            return True
        except Exception as e:
            print(f"Error updating risk parameters: {e}")
            return False
    
    def add_trading_symbol(self, symbol: str, category: str = 'primary') -> bool:
        """Add a new trading symbol"""
        try:
            current_symbols = self.get(f'symbols.{category}', [])
            if symbol not in current_symbols:
                current_symbols.append(symbol)
                self.set(f'symbols.{category}', current_symbols)
            return True
        except Exception as e:
            print(f"Error adding trading symbol: {e}")
            return False
    
    def remove_trading_symbol(self, symbol: str) -> bool:
        """Remove a trading symbol from all categories"""
        try:
            categories = ['primary', 'secondary', 'sectors', 'crypto', 'forex']
            for category in categories:
                symbols = self.get(f'symbols.{category}', [])
                if symbol in symbols:
                    symbols.remove(symbol)
                    self.set(f'symbols.{category}', symbols)
            return True
        except Exception as e:
            print(f"Error removing trading symbol: {e}")
            return False
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate current configuration and return validation results"""
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        try:
            # Check critical parameters
            initial_capital = self.get('trading.initial_capital', 0)
            if initial_capital <= 0:
                validation_results['errors'].append("Initial capital must be positive")
                validation_results['valid'] = False
            
            # Check position size limits
            max_position_size = self.get('trading.max_position_size', 0)
            if max_position_size <= 0 or max_position_size > 1:
                validation_results['errors'].append("Max position size must be between 0 and 1")
                validation_results['valid'] = False
            
            # Check risk parameters
            max_daily_loss = self.get('risk_management.max_daily_loss', 0)
            if max_daily_loss <= 0 or max_daily_loss > 0.5:
                validation_results['warnings'].append("Max daily loss should be between 0 and 50%")
            
            # Check trading symbols
            symbols = self.get_trading_symbols()
            if not symbols:
                validation_results['errors'].append("No trading symbols configured")
                validation_results['valid'] = False
            
            # Check API keys
            openai_key = self.get_api_key('openai')
            if not openai_key:
                validation_results['warnings'].append("OpenAI API key not configured - AI features will be limited")
            
            # Check strategy weights
            total_weight = 0
            for strategy in self.get('strategies', {}):
                if self.is_strategy_enabled(strategy):
                    weight = self.get(f'strategies.{strategy}.weight', 0)
                    total_weight += weight
            
            if abs(total_weight - 1.0) > 0.01:  # Allow small rounding errors
                validation_results['warnings'].append(f"Strategy weights sum to {total_weight:.2f}, should be 1.0")
            
        except Exception as e:
            validation_results['errors'].append(f"Configuration validation error: {e}")
            validation_results['valid'] = False
        
        return validation_results
    
    def export_config(self, export_file: str) -> bool:
        """Export current configuration to a file"""
        try:
            export_data = {
                'config': self._config,
                'export_timestamp': datetime.now().isoformat(),
                'version': '1.0'
            }
            
            with open(export_file, 'w') as f:
                json.dump(export_data, f, indent=4, default=str)
            
            return True
        except Exception as e:
            print(f"Error exporting config: {e}")
            return False
    
    def import_config(self, import_file: str, merge: bool = True) -> bool:
        """Import configuration from a file"""
        try:
            with open(import_file, 'r') as f:
                import_data = json.load(f)
            
            if 'config' not in import_data:
                print("Invalid config file format")
                return False
            
            imported_config = import_data['config']
            
            if merge:
                # Merge with existing config
                self._config.update(imported_config)
            else:
                # Replace entire config
                self._config = imported_config
            
            # Validate and save
            self._validate_config()
            return True
            
        except Exception as e:
            print(f"Error importing config: {e}")
            return False
    
    def reset_to_defaults(self) -> bool:
        """Reset configuration to default values"""
        try:
            self._config = self._get_default_config()
            self._save_config(self._config)
            return True
        except Exception as e:
            print(f"Error resetting config: {e}")
            return False
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of current configuration"""
        return {
            'trading_symbols_count': len(self.get_trading_symbols()),
            'enabled_strategies': [s for s in self.get('strategies', {}) if self.is_strategy_enabled(s)],
            'initial_capital': self.get('trading.initial_capital'),
            'max_position_size': self.get('trading.max_position_size'),
            'max_daily_loss': self.get('risk_management.max_daily_loss'),
            'api_keys_configured': [k.replace('_api_key', '') for k, v in self.get('api_keys', {}).items() if v],
            'last_modified': datetime.fromtimestamp(os.path.getmtime(self.config_file)).isoformat() if os.path.exists(self.config_file) else None
        }
