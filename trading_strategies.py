import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import threading
import time

class TradingSignal:
    """Represents a trading signal"""
    
    def __init__(self, symbol: str, action: str, confidence: float, 
                 price: float, quantity: int, strategy: str, timestamp: datetime = None):
        self.symbol = symbol
        self.action = action  # 'BUY', 'SELL', 'HOLD'
        self.confidence = confidence  # 0-1
        self.price = price
        self.quantity = quantity
        self.strategy = strategy
        self.timestamp = timestamp or datetime.now()

class StrategyManager:
    """Manages multiple trading strategies and generates signals"""
    
    def __init__(self, config, logger, ml_models, ai_advisor):
        self.config = config
        self.logger = logger
        self.ml_models = ml_models
        self.ai_advisor = ai_advisor
        self.active_strategies = {}
        self.strategy_performance = {}
        self.signals_history = []
        
        # Initialize strategies
        self._initialize_strategies()
    
    def _initialize_strategies(self):
        """Initialize all trading strategies"""
        try:
            self.active_strategies = {
                'neural_momentum': NeuralMomentumStrategy(self.config, self.logger, self.ml_models),
                'ensemble_trend': EnsembleTrendStrategy(self.config, self.logger, self.ml_models),
                'ai_pattern': AIPatternRecognitionStrategy(self.config, self.logger, self.ai_advisor),
                'adaptive_mean_reversion': AdaptiveMeanReversionStrategy(self.config, self.logger, self.ml_models),
                'multi_factor': MultiFactorMLStrategy(self.config, self.logger, self.ml_models)
            }
            
            self.logger.info(f"Initialized {len(self.active_strategies)} trading strategies")
            
        except Exception as e:
            self.logger.error(f"Error initializing strategies: {str(e)}")
    
    def generate_signals(self) -> List[TradingSignal]:
        """Generate trading signals from all active strategies"""
        all_signals = []
        
        try:
            symbols = self.config.get_trading_symbols()
            
            for symbol in symbols:
                # Get signals from each strategy
                for strategy_name, strategy in self.active_strategies.items():
                    try:
                        signals = strategy.generate_signal(symbol)
                        if signals:
                            if isinstance(signals, list):
                                all_signals.extend(signals)
                            else:
                                all_signals.append(signals)
                    except Exception as e:
                        self.logger.error(f"Error generating signal from {strategy_name} for {symbol}: {str(e)}")
                        continue
            
            # Filter and rank signals
            filtered_signals = self._filter_and_rank_signals(all_signals)
            
            # Store signals in history
            self.signals_history.extend(filtered_signals)
            
            # Keep only last 1000 signals
            if len(self.signals_history) > 1000:
                self.signals_history = self.signals_history[-1000:]
            
            self.logger.info(f"Generated {len(filtered_signals)} trading signals")
            return filtered_signals
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {str(e)}")
            return []
    
    def _filter_and_rank_signals(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """Filter and rank signals by confidence and other criteria"""
        try:
            if not signals:
                return []
            
            # Filter by minimum confidence
            min_confidence = self.config.get('min_signal_confidence', 0.6)
            filtered = [s for s in signals if s.confidence >= min_confidence]
            
            # Group by symbol and action
            signal_groups = {}
            for signal in filtered:
                key = (signal.symbol, signal.action)
                if key not in signal_groups:
                    signal_groups[key] = []
                signal_groups[key].append(signal)
            
            # For each group, take the highest confidence signal
            final_signals = []
            for group_signals in signal_groups.values():
                best_signal = max(group_signals, key=lambda x: x.confidence)
                final_signals.append(best_signal)
            
            # Sort by confidence (highest first)
            final_signals.sort(key=lambda x: x.confidence, reverse=True)
            
            # Limit number of signals
            max_signals = self.config.get('max_signals_per_cycle', 10)
            return final_signals[:max_signals]
            
        except Exception as e:
            self.logger.error(f"Error filtering signals: {str(e)}")
            return signals
    
    def get_active_strategies(self) -> List[Dict]:
        """Get status of all active strategies"""
        try:
            strategies = []
            
            for name, strategy in self.active_strategies.items():
                performance = self.strategy_performance.get(name, {})
                
                strategies.append({
                    'name': name,
                    'status': 'active' if strategy.is_active else 'inactive',
                    'pnl': performance.get('total_pnl', 0),
                    'win_rate': performance.get('win_rate', 0),
                    'total_trades': performance.get('total_trades', 0),
                    'last_signal': performance.get('last_signal_time', 'Never')
                })
            
            return strategies
            
        except Exception as e:
            self.logger.error(f"Error getting active strategies: {str(e)}")
            return []
    
    def get_strategy_performance(self) -> List[Dict]:
        """Get detailed performance metrics for all strategies"""
        try:
            performance_data = []
            
            for strategy_name, performance in self.strategy_performance.items():
                performance_data.append({
                    'strategy': strategy_name,
                    'total_pnl': performance.get('total_pnl', 0),
                    'win_rate': performance.get('win_rate', 0),
                    'total_trades': performance.get('total_trades', 0),
                    'avg_trade_pnl': performance.get('avg_trade_pnl', 0),
                    'max_drawdown': performance.get('max_drawdown', 0),
                    'sharpe_ratio': performance.get('sharpe_ratio', 0)
                })
            
            return performance_data
            
        except Exception as e:
            self.logger.error(f"Error getting strategy performance: {str(e)}")
            return []
    
    def update_strategy_performance(self, strategy_name: str, trade_result: Dict):
        """Update performance metrics for a strategy"""
        try:
            if strategy_name not in self.strategy_performance:
                self.strategy_performance[strategy_name] = {
                    'total_pnl': 0,
                    'total_trades': 0,
                    'winning_trades': 0,
                    'trade_pnls': [],
                    'last_signal_time': datetime.now()
                }
            
            perf = self.strategy_performance[strategy_name]
            
            # Update basic metrics
            perf['total_pnl'] += trade_result.get('pnl', 0)
            perf['total_trades'] += 1
            perf['trade_pnls'].append(trade_result.get('pnl', 0))
            
            if trade_result.get('pnl', 0) > 0:
                perf['winning_trades'] += 1
            
            # Calculate derived metrics
            perf['win_rate'] = (perf['winning_trades'] / perf['total_trades']) * 100
            perf['avg_trade_pnl'] = perf['total_pnl'] / perf['total_trades']
            
            # Calculate max drawdown
            if len(perf['trade_pnls']) > 1:
                cumulative = np.cumsum(perf['trade_pnls'])
                running_max = np.maximum.accumulate(cumulative)
                drawdown = cumulative - running_max
                perf['max_drawdown'] = abs(min(drawdown))
            
            # Calculate Sharpe ratio (simplified)
            if len(perf['trade_pnls']) > 10:
                returns = np.array(perf['trade_pnls'])
                perf['sharpe_ratio'] = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            
        except Exception as e:
            self.logger.error(f"Error updating strategy performance: {str(e)}")

class BaseStrategy:
    """Base class for all trading strategies"""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.is_active = True
        self.last_signal_time = None
    
    def generate_signal(self, symbol: str) -> Optional[TradingSignal]:
        """Generate trading signal for a symbol"""
        raise NotImplementedError("Subclasses must implement generate_signal method")

class NeuralMomentumStrategy(BaseStrategy):
    """Neural network-based momentum strategy"""
    
    def __init__(self, config, logger, ml_models):
        super().__init__(config, logger)
        self.ml_models = ml_models
        self.lookback_period = 20
    
    def generate_signal(self, symbol: str) -> Optional[TradingSignal]:
        """Generate momentum signal using neural network predictions"""
        try:
            from trading_system.data_processor import DataProcessor
            data_processor = DataProcessor(self.config, self.logger)
            
            # Get recent market data
            data = data_processor.get_market_data(symbol, period="5d", interval="1h")
            if data is None or len(data) < self.lookback_period:
                return None
            
            # Get ML prediction
            features = data_processor.get_features_for_ml(symbol)
            if features is None:
                return None
            
            prediction = self.ml_models.get_ensemble_prediction(symbol, features)
            if prediction is None:
                return None
            
            # Calculate momentum indicators
            recent_return = data['Close'].pct_change(5).iloc[-1]
            rsi = data['RSI'].iloc[-1]
            volume_ratio = data['Volume_Ratio'].iloc[-1]
            
            # Generate signal based on prediction and momentum
            confidence = 0.5
            action = 'HOLD'
            
            if prediction > 0.01 and recent_return > 0 and rsi < 70:
                action = 'BUY'
                confidence = min(0.9, 0.5 + abs(prediction) + (recent_return * 2))
            elif prediction < -0.01 and recent_return < 0 and rsi > 30:
                action = 'SELL'
                confidence = min(0.9, 0.5 + abs(prediction) + (abs(recent_return) * 2))
            
            if action != 'HOLD':
                # Calculate position size based on confidence and volatility
                volatility = data['Volatility'].iloc[-1]
                base_quantity = 100
                quantity = int(base_quantity * confidence / max(0.01, volatility))
                
                signal = TradingSignal(
                    symbol=symbol,
                    action=action,
                    confidence=confidence,
                    price=data['Close'].iloc[-1],
                    quantity=quantity,
                    strategy='neural_momentum'
                )
                
                self.last_signal_time = datetime.now()
                return signal
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in NeuralMomentumStrategy for {symbol}: {str(e)}")
            return None

class EnsembleTrendStrategy(BaseStrategy):
    """Ensemble model-based trend following strategy"""
    
    def __init__(self, config, logger, ml_models):
        super().__init__(config, logger)
        self.ml_models = ml_models
    
    def generate_signal(self, symbol: str) -> Optional[TradingSignal]:
        """Generate trend signal using ensemble predictions"""
        try:
            from trading_system.data_processor import DataProcessor
            data_processor = DataProcessor(self.config, self.logger)
            
            data = data_processor.get_market_data(symbol, period="10d", interval="1h")
            if data is None or len(data) < 50:
                return None
            
            # Calculate trend indicators
            sma_20 = data['SMA_20'].iloc[-1]
            sma_50 = data['SMA_50'].iloc[-1]
            current_price = data['Close'].iloc[-1]
            macd = data['MACD'].iloc[-1]
            macd_signal = data['MACD_Signal'].iloc[-1]
            
            # Get ML ensemble prediction
            features = data_processor.get_features_for_ml(symbol)
            ensemble_pred = self.ml_models.get_ensemble_prediction(symbol, features)
            model_confidence = self.ml_models.get_model_confidence(symbol)
            
            # Trend analysis
            trend_score = 0
            
            # Price vs moving averages
            if current_price > sma_20 > sma_50:
                trend_score += 1
            elif current_price < sma_20 < sma_50:
                trend_score -= 1
            
            # MACD trend
            if macd > macd_signal and macd > 0:
                trend_score += 1
            elif macd < macd_signal and macd < 0:
                trend_score -= 1
            
            # ML prediction
            if ensemble_pred and ensemble_pred > 0.005:
                trend_score += 1
            elif ensemble_pred and ensemble_pred < -0.005:
                trend_score -= 1
            
            # Generate signal
            if trend_score >= 2:
                action = 'BUY'
                confidence = min(0.9, 0.6 + (trend_score * 0.1) + (model_confidence * 0.2))
            elif trend_score <= -2:
                action = 'SELL'
                confidence = min(0.9, 0.6 + (abs(trend_score) * 0.1) + (model_confidence * 0.2))
            else:
                return None
            
            quantity = max(50, int(100 * confidence))
            
            signal = TradingSignal(
                symbol=symbol,
                action=action,
                confidence=confidence,
                price=current_price,
                quantity=quantity,
                strategy='ensemble_trend'
            )
            
            self.last_signal_time = datetime.now()
            return signal
            
        except Exception as e:
            self.logger.error(f"Error in EnsembleTrendStrategy for {symbol}: {str(e)}")
            return None

class AIPatternRecognitionStrategy(BaseStrategy):
    """AI-powered pattern recognition strategy"""
    
    def __init__(self, config, logger, ai_advisor):
        super().__init__(config, logger)
        self.ai_advisor = ai_advisor
    
    def generate_signal(self, symbol: str) -> Optional[TradingSignal]:
        """Generate signal based on AI pattern recognition"""
        try:
            from trading_system.data_processor import DataProcessor
            data_processor = DataProcessor(self.config, self.logger)
            
            data = data_processor.get_market_data(symbol, period="7d", interval="1h")
            if data is None or len(data) < 50:
                return None
            
            # Get AI analysis of current market patterns
            ai_signal = self.ai_advisor.analyze_trading_opportunity(symbol, data)
            
            if ai_signal and ai_signal.get('action') != 'HOLD':
                signal = TradingSignal(
                    symbol=symbol,
                    action=ai_signal['action'],
                    confidence=ai_signal.get('confidence', 0.5),
                    price=data['Close'].iloc[-1],
                    quantity=ai_signal.get('quantity', 100),
                    strategy='ai_pattern'
                )
                
                self.last_signal_time = datetime.now()
                return signal
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in AIPatternRecognitionStrategy for {symbol}: {str(e)}")
            return None

class AdaptiveMeanReversionStrategy(BaseStrategy):
    """Adaptive mean reversion strategy using ML"""
    
    def __init__(self, config, logger, ml_models):
        super().__init__(config, logger)
        self.ml_models = ml_models
    
    def generate_signal(self, symbol: str) -> Optional[TradingSignal]:
        """Generate mean reversion signal"""
        try:
            from trading_system.data_processor import DataProcessor
            data_processor = DataProcessor(self.config, self.logger)
            
            data = data_processor.get_market_data(symbol, period="5d", interval="30m")
            if data is None or len(data) < 50:
                return None
            
            # Calculate mean reversion indicators
            current_price = data['Close'].iloc[-1]
            bb_upper = data['BB_Upper'].iloc[-1]
            bb_lower = data['BB_Lower'].iloc[-1]
            bb_middle = data['BB_Middle'].iloc[-1]
            rsi = data['RSI'].iloc[-1]
            
            # Z-score for mean reversion
            price_sma = data['Close'].rolling(20).mean().iloc[-1]
            price_std = data['Close'].rolling(20).std().iloc[-1]
            z_score = (current_price - price_sma) / price_std if price_std > 0 else 0
            
            # Generate mean reversion signal
            action = 'HOLD'
            confidence = 0.5
            
            # Oversold conditions (buy signal)
            if (current_price < bb_lower and rsi < 30 and z_score < -1.5):
                action = 'BUY'
                confidence = min(0.9, 0.6 + abs(z_score) * 0.1 + (30 - rsi) * 0.01)
            
            # Overbought conditions (sell signal)
            elif (current_price > bb_upper and rsi > 70 and z_score > 1.5):
                action = 'SELL'
                confidence = min(0.9, 0.6 + abs(z_score) * 0.1 + (rsi - 70) * 0.01)
            
            if action != 'HOLD':
                quantity = max(25, int(75 * confidence))
                
                signal = TradingSignal(
                    symbol=symbol,
                    action=action,
                    confidence=confidence,
                    price=current_price,
                    quantity=quantity,
                    strategy='adaptive_mean_reversion'
                )
                
                self.last_signal_time = datetime.now()
                return signal
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in AdaptiveMeanReversionStrategy for {symbol}: {str(e)}")
            return None

class MultiFactorMLStrategy(BaseStrategy):
    """Multi-factor machine learning strategy"""
    
    def __init__(self, config, logger, ml_models):
        super().__init__(config, logger)
        self.ml_models = ml_models
    
    def generate_signal(self, symbol: str) -> Optional[TradingSignal]:
        """Generate signal using multiple ML factors"""
        try:
            from trading_system.data_processor import DataProcessor
            data_processor = DataProcessor(self.config, self.logger)
            
            data = data_processor.get_market_data(symbol, period="10d", interval="1h")
            if data is None or len(data) < 100:
                return None
            
            # Get predictions from all ML models
            features = data_processor.get_features_for_ml(symbol)
            predictions = self.ml_models.predict(symbol, features)
            
            if not predictions:
                return None
            
            # Calculate factor scores
            factors = {
                'momentum': self._calculate_momentum_factor(data),
                'value': self._calculate_value_factor(data),
                'quality': self._calculate_quality_factor(data),
                'volatility': self._calculate_volatility_factor(data)
            }
            
            # Combine ML predictions with factors
            ml_score = np.mean(list(predictions.values()))
            factor_score = np.mean(list(factors.values()))
            
            combined_score = (ml_score * 0.6) + (factor_score * 0.4)
            
            # Generate signal
            if combined_score > 0.02:
                action = 'BUY'
                confidence = min(0.9, 0.5 + abs(combined_score) * 10)
            elif combined_score < -0.02:
                action = 'SELL'
                confidence = min(0.9, 0.5 + abs(combined_score) * 10)
            else:
                return None
            
            quantity = max(50, int(100 * confidence))
            
            signal = TradingSignal(
                symbol=symbol,
                action=action,
                confidence=confidence,
                price=data['Close'].iloc[-1],
                quantity=quantity,
                strategy='multi_factor_ml'
            )
            
            self.last_signal_time = datetime.now()
            return signal
            
        except Exception as e:
            self.logger.error(f"Error in MultiFactorMLStrategy for {symbol}: {str(e)}")
            return None
    
    def _calculate_momentum_factor(self, data: pd.DataFrame) -> float:
        """Calculate momentum factor score"""
        try:
            # 20-day momentum
            momentum_20 = data['Close'].pct_change(20).iloc[-1]
            return np.tanh(momentum_20 * 10)  # Normalize to [-1, 1]
        except:
            return 0.0
    
    def _calculate_value_factor(self, data: pd.DataFrame) -> float:
        """Calculate value factor score (simplified)"""
        try:
            # Price relative to moving average
            current_price = data['Close'].iloc[-1]
            sma_50 = data['SMA_50'].iloc[-1]
            value_score = (sma_50 - current_price) / current_price
            return np.tanh(value_score * 5)
        except:
            return 0.0
    
    def _calculate_quality_factor(self, data: pd.DataFrame) -> float:
        """Calculate quality factor score"""
        try:
            # Volume trend and price stability
            volume_trend = data['Volume'].pct_change(10).iloc[-1]
            volatility = data['Volatility'].iloc[-1]
            quality_score = volume_trend - volatility
            return np.tanh(quality_score * 5)
        except:
            return 0.0
    
    def _calculate_volatility_factor(self, data: pd.DataFrame) -> float:
        """Calculate volatility factor score"""
        try:
            # Relative volatility
            current_vol = data['Volatility'].iloc[-1]
            avg_vol = data['Volatility'].rolling(50).mean().iloc[-1]
            vol_score = (avg_vol - current_vol) / avg_vol if avg_vol > 0 else 0
            return np.tanh(vol_score * 3)
        except:
            return 0.0
