import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
# TensorFlow imports temporarily disabled due to compatibility issues
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class MLModelManager:
    """Manages multiple machine learning models for trading predictions"""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.models = {}
        self.scalers = {}
        self.model_performance = {}
        self.is_trained = {}
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize different ML models"""
        try:
            # Random Forest for trend prediction
            self.models['random_forest'] = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            # Gradient Boosting for price prediction
            self.models['gradient_boosting'] = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            
            # Neural Network for pattern recognition
            self.models['neural_network'] = MLPRegressor(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                max_iter=500,
                random_state=42
            )
            
            # Initialize scalers for each model
            for model_name in self.models.keys():
                self.scalers[model_name] = StandardScaler()
                self.is_trained[model_name] = False
            
            self.logger.info("ML models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing ML models: {str(e)}")
    
    def create_lstm_model(self, input_shape: Tuple[int, int]):
        """Create LSTM model for time series prediction - Currently disabled due to TensorFlow compatibility issues"""
        try:
            self.logger.warning("LSTM model creation temporarily disabled due to TensorFlow compatibility issues")
            return None
            # Commented out due to TensorFlow issues:
            # model = Sequential([
            #     LSTM(50, return_sequences=True, input_shape=input_shape),
            #     Dropout(0.2),
            #     LSTM(50, return_sequences=False),
            #     Dropout(0.2),
            #     Dense(25),
            #     Dense(1)
            # ])
            # 
            # model.compile(
            #     optimizer='adam',
            #     loss='mean_squared_error',
            #     metrics=['mae']
            # )
            # 
            # return model
            
        except Exception as e:
            self.logger.error(f"Error creating LSTM model: {str(e)}")
            return None
    
    def prepare_training_data(self, data: pd.DataFrame, target_column: str = 'Close', 
                            lookback: int = 60) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepare data for training"""
        try:
            # Remove rows with NaN values
            data = data.dropna()
            
            if len(data) < lookback + 1:
                self.logger.warning("Insufficient data for training")
                return None, None
            
            # Create features (all columns except target)
            feature_columns = [col for col in data.columns if col != target_column and not col.endswith('_norm')]
            X = data[feature_columns].values
            y = data[target_column].values
            
            # Create sequences for time series
            X_sequences = []
            y_sequences = []
            
            for i in range(lookback, len(data)):
                X_sequences.append(X[i-lookback:i])
                y_sequences.append(y[i])
            
            return np.array(X_sequences), np.array(y_sequences)
            
        except Exception as e:
            self.logger.error(f"Error preparing training data: {str(e)}")
            return None, None
    
    def train_models(self, symbol: str, data: pd.DataFrame) -> Dict[str, float]:
        """Train all models on the provided data"""
        training_results = {}
        
        try:
            # Prepare different target variables
            targets = {
                'price_direction': self._create_direction_target(data),
                'price_return': data['Close'].pct_change().shift(-1),
                'volatility': data['Close'].rolling(window=20).std()
            }
            
            for target_name, target_values in targets.items():
                if target_values is None:
                    continue
                    
                # Prepare features
                feature_columns = [
                    'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
                    'MACD', 'MACD_Signal', 'RSI',
                    'Volume_Ratio', 'High_Low_Ratio', 'Volatility'
                ]
                
                # Filter available columns
                available_features = [col for col in feature_columns if col in data.columns]
                
                if not available_features:
                    continue
                
                X = data[available_features].dropna()
                y = target_values[X.index]
                
                if len(X) < 50:  # Minimum data requirement
                    continue
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, shuffle=False
                )
                
                # Train traditional ML models
                for model_name, model in self.models.items():
                    try:
                        # Scale features
                        scaler_key = f"{model_name}_{target_name}"
                        if scaler_key not in self.scalers:
                            self.scalers[scaler_key] = StandardScaler()
                        
                        X_train_scaled = self.scalers[scaler_key].fit_transform(X_train)
                        X_test_scaled = self.scalers[scaler_key].transform(X_test)
                        
                        # Train model
                        model.fit(X_train_scaled, y_train)
                        
                        # Evaluate
                        y_pred = model.predict(X_test_scaled)
                        score = r2_score(y_test, y_pred)
                        
                        training_results[f"{model_name}_{target_name}"] = score
                        self.is_trained[model_name] = True
                        
                        self.logger.info(f"Trained {model_name} for {target_name}: R² = {score:.4f}")
                        
                    except Exception as e:
                        self.logger.error(f"Error training {model_name} for {target_name}: {str(e)}")
                        continue
            
            # Train LSTM model
            try:
                X_lstm, y_lstm = self.prepare_training_data(data)
                if X_lstm is not None and y_lstm is not None:
                    lstm_score = self._train_lstm_model(X_lstm, y_lstm)
                    if lstm_score:
                        training_results['lstm'] = lstm_score
            except Exception as e:
                self.logger.error(f"Error training LSTM model: {str(e)}")
            
            # Store performance
            self.model_performance[symbol] = training_results
            
            return training_results
            
        except Exception as e:
            self.logger.error(f"Error training models for {symbol}: {str(e)}")
            return {}
    
    def _create_direction_target(self, data: pd.DataFrame) -> Optional[pd.Series]:
        """Create binary target for price direction prediction"""
        try:
            # 1 if price goes up, 0 if price goes down
            future_return = data['Close'].shift(-1) / data['Close'] - 1
            return (future_return > 0).astype(int)
        except Exception as e:
            self.logger.error(f"Error creating direction target: {str(e)}")
            return None
    
    def _train_lstm_model(self, X: np.ndarray, y: np.ndarray) -> Optional[float]:
        """Train LSTM model - Currently disabled due to TensorFlow compatibility issues"""
        try:
            self.logger.warning("LSTM training temporarily disabled due to TensorFlow compatibility issues")
            return None
            
            # Commented out due to TensorFlow issues:
            # # Reshape for LSTM (samples, time steps, features)
            # if len(X.shape) != 3:
            #     return None
            # 
            # # Split data
            # split_idx = int(0.8 * len(X))
            # X_train, X_test = X[:split_idx], X[split_idx:]
            # y_train, y_test = y[:split_idx], y[split_idx:]
            # 
            # # Create and train model
            # model = self.create_lstm_model((X.shape[1], X.shape[2]))
            # if model is None:
            #     return None
            # 
            # # Train with early stopping
            # model.fit(
            #     X_train, y_train,
            #     epochs=50,
            #     batch_size=32,
            #     validation_split=0.2,
            #     verbose=0
            # )
            # 
            # # Evaluate
            # y_pred = model.predict(X_test)
            # score = r2_score(y_test, y_pred)
            # 
            # # Store trained model
            # self.models['lstm'] = model
            # self.is_trained['lstm'] = True
            # 
            # return score
            
        except Exception as e:
            self.logger.error(f"Error training LSTM model: {str(e)}")
            return None
    
    def predict(self, symbol: str, features: pd.DataFrame) -> Dict[str, float]:
        """Generate predictions from all trained models"""
        predictions = {}
        
        try:
            # Prepare features for traditional ML models
            feature_columns = [
                'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
                'MACD', 'MACD_Signal', 'RSI',
                'Volume_Ratio', 'High_Low_Ratio', 'Volatility'
            ]
            
            available_features = [col for col in feature_columns if col in features.columns]
            
            if not available_features:
                return predictions
            
            X = features[available_features].iloc[-1:].values
            
            # Get predictions from each trained model
            for model_name, model in self.models.items():
                if not self.is_trained.get(model_name, False):
                    continue
                
                try:
                    if model_name == 'lstm':
                        # LSTM temporarily disabled due to TensorFlow compatibility issues
                        continue
                    else:
                        # Traditional ML models
                        scaler_key = f"{model_name}_price_return"
                        if scaler_key in self.scalers:
                            X_scaled = self.scalers[scaler_key].transform(X)
                            pred = model.predict(X_scaled)[0]
                            predictions[model_name] = float(pred)
                            
                except Exception as e:
                    self.logger.error(f"Error getting prediction from {model_name}: {str(e)}")
                    continue
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error generating predictions for {symbol}: {str(e)}")
            return {}
    
    def get_ensemble_prediction(self, symbol: str, features: pd.DataFrame) -> Optional[float]:
        """Get weighted ensemble prediction from all models"""
        try:
            predictions = self.predict(symbol, features)
            
            if not predictions:
                return None
            
            # Weight predictions by model performance
            weights = {}
            total_weight = 0
            
            for model_name, prediction in predictions.items():
                # Get model performance (default to 0.5 if not available)
                performance_key = f"{model_name}_price_return"
                performance = self.model_performance.get(symbol, {}).get(performance_key, 0.5)
                
                # Convert R² to positive weight (minimum 0.1)
                weight = max(0.1, performance) if performance > 0 else 0.1
                weights[model_name] = weight
                total_weight += weight
            
            # Calculate weighted average
            ensemble_prediction = sum(
                pred * weights[model_name] / total_weight 
                for model_name, pred in predictions.items()
            )
            
            return ensemble_prediction
            
        except Exception as e:
            self.logger.error(f"Error calculating ensemble prediction: {str(e)}")
            return None
    
    def get_model_confidence(self, symbol: str) -> float:
        """Get confidence score based on model performance"""
        try:
            if symbol not in self.model_performance:
                return 0.5
            
            performances = list(self.model_performance[symbol].values())
            if not performances:
                return 0.5
            
            # Average performance as confidence
            avg_performance = np.mean([p for p in performances if p > 0])
            return min(1.0, max(0.0, float(avg_performance)))
            
        except Exception as e:
            self.logger.error(f"Error calculating model confidence: {str(e)}")
            return 0.5
