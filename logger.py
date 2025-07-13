import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
import threading

class TradingLogger:
    """Comprehensive logging system for the trading application"""
    
    def __init__(self, log_dir: str = "logs", max_file_size: int = 10*1024*1024, backup_count: int = 5):
        self.log_dir = log_dir
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self._lock = threading.Lock()
        
        # Create logs directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize loggers
        self._setup_loggers()
    
    def _setup_loggers(self):
        """Set up different loggers for different components"""
        
        # Main application logger
        self.main_logger = self._create_logger(
            'trading_system',
            os.path.join(self.log_dir, 'trading_system.log'),
            logging.INFO
        )
        
        # Trading-specific logger
        self.trading_logger = self._create_logger(
            'trading',
            os.path.join(self.log_dir, 'trading.log'),
            logging.INFO
        )
        
        # Risk management logger
        self.risk_logger = self._create_logger(
            'risk_management',
            os.path.join(self.log_dir, 'risk_management.log'),
            logging.WARNING
        )
        
        # Data processing logger
        self.data_logger = self._create_logger(
            'data_processing',
            os.path.join(self.log_dir, 'data_processing.log'),
            logging.ERROR
        )
        
        # ML models logger
        self.ml_logger = self._create_logger(
            'ml_models',
            os.path.join(self.log_dir, 'ml_models.log'),
            logging.INFO
        )
        
        # Performance logger
        self.performance_logger = self._create_logger(
            'performance',
            os.path.join(self.log_dir, 'performance.log'),
            logging.INFO
        )
        
        # Error logger for critical issues
        self.error_logger = self._create_logger(
            'errors',
            os.path.join(self.log_dir, 'errors.log'),
            logging.ERROR
        )
    
    def _create_logger(self, name: str, log_file: str, level: int) -> logging.Logger:
        """Create a logger with file rotation"""
        
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Remove existing handlers to avoid duplicates
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler with rotation
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=self.max_file_size,
            backupCount=self.backup_count
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        
        # Console handler for important messages
        if level <= logging.WARNING:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            console_handler.setLevel(logging.WARNING)
            logger.addHandler(console_handler)
        
        logger.addHandler(file_handler)
        
        # Prevent propagation to root logger
        logger.propagate = False
        
        return logger
    
    def info(self, message: str, component: str = 'main'):
        """Log info message"""
        with self._lock:
            try:
                logger = self._get_logger(component)
                logger.info(message)
            except Exception as e:
                print(f"Logging error: {e}")
    
    def warning(self, message: str, component: str = 'main'):
        """Log warning message"""
        with self._lock:
            try:
                logger = self._get_logger(component)
                logger.warning(message)
            except Exception as e:
                print(f"Logging error: {e}")
    
    def error(self, message: str, component: str = 'main'):
        """Log error message"""
        with self._lock:
            try:
                logger = self._get_logger(component)
                logger.error(message)
                # Also log to error logger for critical tracking
                self.error_logger.error(f"[{component}] {message}")
            except Exception as e:
                print(f"Logging error: {e}")
    
    def critical(self, message: str, component: str = 'main'):
        """Log critical message"""
        with self._lock:
            try:
                logger = self._get_logger(component)
                logger.critical(message)
                # Also log to error logger
                self.error_logger.critical(f"[{component}] {message}")
            except Exception as e:
                print(f"Logging error: {e}")
    
    def debug(self, message: str, component: str = 'main'):
        """Log debug message"""
        with self._lock:
            try:
                logger = self._get_logger(component)
                logger.debug(message)
            except Exception as e:
                print(f"Logging error: {e}")
    
    def _get_logger(self, component: str) -> logging.Logger:
        """Get appropriate logger based on component"""
        component_map = {
            'main': self.main_logger,
            'trading': self.trading_logger,
            'risk': self.risk_logger,
            'data': self.data_logger,
            'ml': self.ml_logger,
            'performance': self.performance_logger,
            'error': self.error_logger
        }
        
        return component_map.get(component, self.main_logger)
    
    def log_trade(self, trade_info: dict):
        """Log trade execution details"""
        try:
            trade_message = (
                f"TRADE EXECUTED - Symbol: {trade_info.get('symbol', 'Unknown')}, "
                f"Action: {trade_info.get('action', 'Unknown')}, "
                f"Quantity: {trade_info.get('quantity', 0)}, "
                f"Price: ${trade_info.get('price', 0):.2f}, "
                f"Strategy: {trade_info.get('strategy', 'Unknown')}, "
                f"Status: {trade_info.get('status', 'Unknown')}"
            )
            self.trading_logger.info(trade_message)
        except Exception as e:
            self.error(f"Error logging trade: {e}", 'trading')
    
    def log_risk_event(self, risk_info: dict):
        """Log risk management events"""
        try:
            risk_message = (
                f"RISK EVENT - Type: {risk_info.get('type', 'Unknown')}, "
                f"Severity: {risk_info.get('severity', 'Unknown')}, "
                f"Message: {risk_info.get('message', 'No details')}, "
                f"Portfolio Impact: {risk_info.get('portfolio_impact', 'Unknown')}"
            )
            self.risk_logger.warning(risk_message)
        except Exception as e:
            self.error(f"Error logging risk event: {e}", 'risk')
    
    def log_performance(self, performance_data: dict):
        """Log performance metrics"""
        try:
            performance_message = (
                f"PERFORMANCE UPDATE - "
                f"Total Return: {performance_data.get('total_return', 0):.2%}, "
                f"Daily P&L: ${performance_data.get('daily_pnl', 0):.2f}, "
                f"Win Rate: {performance_data.get('win_rate', 0):.1f}%, "
                f"Sharpe Ratio: {performance_data.get('sharpe_ratio', 0):.2f}, "
                f"Max Drawdown: {performance_data.get('max_drawdown', 0):.2%}"
            )
            self.performance_logger.info(performance_message)
        except Exception as e:
            self.error(f"Error logging performance: {e}", 'performance')
    
    def log_ml_training(self, model_info: dict):
        """Log ML model training events"""
        try:
            ml_message = (
                f"ML MODEL TRAINING - "
                f"Model: {model_info.get('model_name', 'Unknown')}, "
                f"Symbol: {model_info.get('symbol', 'Unknown')}, "
                f"Performance Score: {model_info.get('score', 0):.4f}, "
                f"Training Samples: {model_info.get('samples', 0)}, "
                f"Status: {model_info.get('status', 'Unknown')}"
            )
            self.ml_logger.info(ml_message)
        except Exception as e:
            self.error(f"Error logging ML training: {e}", 'ml')
    
    def log_data_update(self, data_info: dict):
        """Log data processing events"""
        try:
            data_message = (
                f"DATA UPDATE - "
                f"Symbol: {data_info.get('symbol', 'Unknown')}, "
                f"Records: {data_info.get('records', 0)}, "
                f"Source: {data_info.get('source', 'Unknown')}, "
                f"Status: {data_info.get('status', 'Unknown')}, "
                f"Latency: {data_info.get('latency_ms', 0)}ms"
            )
            self.data_logger.info(data_message)
        except Exception as e:
            self.error(f"Error logging data update: {e}", 'data')
    
    def log_system_status(self, status_info: dict):
        """Log system status changes"""
        try:
            status_message = (
                f"SYSTEM STATUS - "
                f"Component: {status_info.get('component', 'Unknown')}, "
                f"Status: {status_info.get('status', 'Unknown')}, "
                f"Message: {status_info.get('message', 'No details')}, "
                f"Timestamp: {datetime.now().isoformat()}"
            )
            self.main_logger.info(status_message)
        except Exception as e:
            self.error(f"Error logging system status: {e}", 'main')
    
    def log_ai_analysis(self, ai_info: dict):
        """Log AI analysis and decision events"""
        try:
            ai_message = (
                f"AI ANALYSIS - "
                f"Type: {ai_info.get('analysis_type', 'Unknown')}, "
                f"Symbol: {ai_info.get('symbol', 'Market')}, "
                f"Confidence: {ai_info.get('confidence', 0):.2f}, "
                f"Recommendation: {ai_info.get('recommendation', 'None')}, "
                f"Reasoning: {ai_info.get('reasoning', 'No details provided')}"
            )
            self.main_logger.info(ai_message)
        except Exception as e:
            self.error(f"Error logging AI analysis: {e}", 'main')
    
    def get_recent_logs(self, component: str = 'main', lines: int = 100) -> list:
        """Get recent log entries"""
        try:
            log_files = {
                'main': os.path.join(self.log_dir, 'trading_system.log'),
                'trading': os.path.join(self.log_dir, 'trading.log'),
                'risk': os.path.join(self.log_dir, 'risk_management.log'),
                'data': os.path.join(self.log_dir, 'data_processing.log'),
                'ml': os.path.join(self.log_dir, 'ml_models.log'),
                'performance': os.path.join(self.log_dir, 'performance.log'),
                'error': os.path.join(self.log_dir, 'errors.log')
            }
            
            log_file = log_files.get(component, log_files['main'])
            
            if not os.path.exists(log_file):
                return []
            
            with open(log_file, 'r') as f:
                all_lines = f.readlines()
                return all_lines[-lines:] if len(all_lines) > lines else all_lines
                
        except Exception as e:
            self.error(f"Error getting recent logs: {e}", 'main')
            return []
    
    def clear_old_logs(self, days_to_keep: int = 30):
        """Clear log files older than specified days"""
        try:
            cutoff_time = datetime.now().timestamp() - (days_to_keep * 24 * 3600)
            
            for filename in os.listdir(self.log_dir):
                file_path = os.path.join(self.log_dir, filename)
                if os.path.isfile(file_path):
                    file_time = os.path.getmtime(file_path)
                    if file_time < cutoff_time:
                        os.remove(file_path)
                        self.info(f"Removed old log file: {filename}")
                        
        except Exception as e:
            self.error(f"Error clearing old logs: {e}", 'main')
    
    def get_log_statistics(self) -> dict:
        """Get logging statistics"""
        try:
            stats = {}
            
            for component in ['main', 'trading', 'risk', 'data', 'ml', 'performance', 'error']:
                log_files = {
                    'main': 'trading_system.log',
                    'trading': 'trading.log',
                    'risk': 'risk_management.log',
                    'data': 'data_processing.log',
                    'ml': 'ml_models.log',
                    'performance': 'performance.log',
                    'error': 'errors.log'
                }
                
                log_file = os.path.join(self.log_dir, log_files[component])
                
                if os.path.exists(log_file):
                    file_size = os.path.getsize(log_file)
                    with open(log_file, 'r') as f:
                        line_count = sum(1 for _ in f)
                    
                    stats[component] = {
                        'file_size_bytes': file_size,
                        'line_count': line_count,
                        'last_modified': datetime.fromtimestamp(os.path.getmtime(log_file)).isoformat()
                    }
                else:
                    stats[component] = {
                        'file_size_bytes': 0,
                        'line_count': 0,
                        'last_modified': None
                    }
            
            return stats
            
        except Exception as e:
            self.error(f"Error getting log statistics: {e}", 'main')
            return {}
