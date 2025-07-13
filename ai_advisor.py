import json
import os
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime

# the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# do not change this unless explicitly requested by the user
from openai import OpenAI

class AIAdvisor:
    """AI-powered trading advisor using GPT-4o for market analysis and strategy optimization"""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.openai_client = self._initialize_openai()
        self.analysis_cache = {}
        self.max_cache_age = 300  # 5 minutes
        
    def _initialize_openai(self) -> Optional[OpenAI]:
        """Initialize OpenAI client"""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                self.logger.warning("OpenAI API key not found. AI features will be limited.")
                return None
            
            client = OpenAI(api_key=api_key)
            self.logger.info("OpenAI client initialized successfully")
            return client
            
        except Exception as e:
            self.logger.error(f"Error initializing OpenAI client: {str(e)}")
            return None
    
    def get_market_insights(self) -> List[Dict]:
        """Generate AI-powered market insights"""
        try:
            if not self.openai_client:
                return [{'type': 'Error', 'content': 'OpenAI client not available', 'confidence': 0.0}]
            
            # Check cache
            cache_key = f"market_insights_{datetime.now().strftime('%Y%m%d_%H%M')}"
            if cache_key in self.analysis_cache:
                cached_time, cached_result = self.analysis_cache[cache_key]
                if (datetime.now() - cached_time).seconds < self.max_cache_age:
                    return cached_result
            
            # Get current market data context
            market_context = self._get_market_context()
            
            prompt = f"""
            Analyze the current market conditions and provide 3-5 key trading insights.
            
            Current Market Context:
            {market_context}
            
            Please provide insights in JSON format with the following structure:
            {{
                "insights": [
                    {{
                        "type": "Market Trend",
                        "content": "Brief insight description",
                        "confidence": 0.8,
                        "timeframe": "short-term/medium-term/long-term",
                        "impact": "bullish/bearish/neutral"
                    }}
                ]
            }}
            
            Focus on:
            1. Overall market sentiment and trends
            2. Sector rotation opportunities
            3. Risk factors to watch
            4. Technical patterns observed
            5. Economic indicators impact
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert financial analyst with deep knowledge of market dynamics, technical analysis, and trading strategies. Provide objective, data-driven insights."
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            insights = result.get('insights', [])
            
            # Cache the result
            self.analysis_cache[cache_key] = (datetime.now(), insights)
            
            self.logger.info(f"Generated {len(insights)} AI market insights")
            return insights
            
        except Exception as e:
            self.logger.error(f"Error generating market insights: {str(e)}")
            return [{'type': 'Error', 'content': f'Failed to generate insights: {str(e)}', 'confidence': 0.0}]
    
    def analyze_trading_opportunity(self, symbol: str, market_data: pd.DataFrame) -> Optional[Dict]:
        """Analyze a specific trading opportunity using AI"""
        try:
            if not self.openai_client:
                return None
            
            # Prepare market data summary
            data_summary = self._prepare_data_summary(symbol, market_data)
            
            prompt = f"""
            Analyze the trading opportunity for {symbol} based on the following market data:
            
            {data_summary}
            
            Provide your analysis in JSON format:
            {{
                "action": "BUY/SELL/HOLD",
                "confidence": 0.0-1.0,
                "reasoning": "Brief explanation of the decision",
                "entry_price": estimated_entry_price,
                "stop_loss": suggested_stop_loss_price,
                "take_profit": suggested_take_profit_price,
                "quantity": suggested_position_size,
                "timeframe": "short/medium/long",
                "risk_level": "low/medium/high"
            }}
            
            Consider:
            1. Technical indicators and price action
            2. Volume patterns and momentum
            3. Support and resistance levels
            4. Risk-reward ratio
            5. Current market conditions
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional trading analyst. Analyze the data objectively and provide specific, actionable trading recommendations with clear risk management."
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            analysis = json.loads(response.choices[0].message.content)
            
            # Validate and normalize the response
            if analysis.get('action') not in ['BUY', 'SELL', 'HOLD']:
                analysis['action'] = 'HOLD'
            
            analysis['confidence'] = max(0.0, min(1.0, float(analysis.get('confidence', 0.5))))
            
            self.logger.info(f"AI analysis for {symbol}: {analysis['action']} (confidence: {analysis['confidence']:.2f})")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing trading opportunity for {symbol}: {str(e)}")
            return None
    
    def optimize_strategy_parameters(self, strategy_name: str, performance_data: Dict) -> Dict:
        """Use AI to optimize strategy parameters based on performance"""
        try:
            if not self.openai_client:
                return {}
            
            prompt = f"""
            Optimize the parameters for the {strategy_name} trading strategy based on its performance data:
            
            Performance Data:
            {json.dumps(performance_data, indent=2)}
            
            Suggest parameter optimizations in JSON format:
            {{
                "optimizations": [
                    {{
                        "parameter": "parameter_name",
                        "current_value": current_value,
                        "suggested_value": suggested_value,
                        "reasoning": "Why this change is recommended"
                    }}
                ],
                "overall_assessment": "Strategy performance assessment",
                "confidence": 0.0-1.0
            }}
            
            Focus on:
            1. Risk-adjusted returns improvement
            2. Drawdown reduction
            3. Win rate optimization
            4. Position sizing adjustments
            5. Entry/exit timing refinements
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a quantitative analyst specializing in trading strategy optimization. Provide data-driven recommendations for improving strategy performance."
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            optimization = json.loads(response.choices[0].message.content)
            
            self.logger.info(f"Generated AI optimization for {strategy_name}")
            return optimization
            
        except Exception as e:
            self.logger.error(f"Error optimizing strategy {strategy_name}: {str(e)}")
            return {}
    
    def generate_risk_assessment(self, portfolio_data: Dict, market_conditions: Dict) -> Dict:
        """Generate AI-powered risk assessment"""
        try:
            if not self.openai_client:
                return {'risk_level': 'medium', 'recommendations': []}
            
            prompt = f"""
            Assess the risk profile of this trading portfolio:
            
            Portfolio Data:
            {json.dumps(portfolio_data, indent=2)}
            
            Market Conditions:
            {json.dumps(market_conditions, indent=2)}
            
            Provide risk assessment in JSON format:
            {{
                "overall_risk_level": "low/medium/high/critical",
                "risk_score": 0-10,
                "key_risks": [
                    {{
                        "type": "risk_category",
                        "description": "specific_risk_description",
                        "severity": "low/medium/high",
                        "probability": 0.0-1.0
                    }}
                ],
                "recommendations": [
                    "Specific action recommendation"
                ],
                "confidence": 0.0-1.0
            }}
            
            Analyze:
            1. Position concentration risk
            2. Market exposure levels
            3. Correlation risks
            4. Liquidity concerns
            5. Volatility exposure
            6. Drawdown potential
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a risk management expert with extensive experience in portfolio risk analysis. Provide comprehensive and actionable risk assessments."
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            risk_assessment = json.loads(response.choices[0].message.content)
            
            self.logger.info(f"Generated AI risk assessment: {risk_assessment.get('overall_risk_level', 'unknown')} risk")
            return risk_assessment
            
        except Exception as e:
            self.logger.error(f"Error generating risk assessment: {str(e)}")
            return {'risk_level': 'medium', 'recommendations': []}
    
    def _get_market_context(self) -> str:
        """Get current market context for AI analysis"""
        try:
            # This would gather real market data
            # For now, provide a structured placeholder
            context = """
            Current Market Environment:
            - Major indices: Mixed performance with tech leading
            - Volatility: Moderate levels (VIX around 18-22)
            - Interest rates: Current Fed funds rate environment
            - Economic indicators: Recent employment, inflation data
            - Sector performance: Technology and healthcare outperforming
            - Global factors: Geopolitical considerations, currency movements
            
            Recent Market Events:
            - Federal Reserve policy decisions
            - Corporate earnings trends
            - Economic data releases
            - Global market developments
            """
            
            return context
            
        except Exception as e:
            self.logger.error(f"Error getting market context: {str(e)}")
            return "Market context unavailable"
    
    def _prepare_data_summary(self, symbol: str, data: pd.DataFrame) -> str:
        """Prepare market data summary for AI analysis"""
        try:
            if data.empty:
                return f"No data available for {symbol}"
            
            latest = data.iloc[-1]
            
            summary = f"""
            Symbol: {symbol}
            Current Price: ${latest['Close']:.2f}
            
            Technical Indicators:
            - 20-day SMA: ${latest.get('SMA_20', 0):.2f}
            - 50-day SMA: ${latest.get('SMA_50', 0):.2f}
            - RSI: {latest.get('RSI', 0):.1f}
            - MACD: {latest.get('MACD', 0):.3f}
            - Volume Ratio: {latest.get('Volume_Ratio', 0):.2f}
            
            Recent Performance:
            - 1-day change: {data['Close'].pct_change(1).iloc[-1]:.2%}
            - 5-day change: {data['Close'].pct_change(5).iloc[-1]:.2%}
            - 20-day volatility: {data['Close'].pct_change().rolling(20).std().iloc[-1]:.2%}
            
            Price Action:
            - High/Low ratio: {latest.get('High_Low_Ratio', 0):.3f}
            - Bollinger Bands position: Price vs BB_Upper: {(latest['Close'] / latest.get('BB_Upper', latest['Close'])):.3f}
            
            Volume Analysis:
            - Current volume vs average: {latest.get('Volume_Ratio', 1):.2f}x
            """
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error preparing data summary for {symbol}: {str(e)}")
            return f"Error analyzing data for {symbol}"
    
    def get_portfolio_optimization_suggestions(self, current_portfolio: Dict, 
                                            market_outlook: str = "neutral") -> List[Dict]:
        """Get AI suggestions for portfolio optimization"""
        try:
            if not self.openai_client:
                return []
            
            prompt = f"""
            Provide portfolio optimization suggestions based on the current holdings and market outlook:
            
            Current Portfolio:
            {json.dumps(current_portfolio, indent=2)}
            
            Market Outlook: {market_outlook}
            
            Provide suggestions in JSON format:
            {{
                "suggestions": [
                    {{
                        "type": "rebalance/add/reduce/hedge",
                        "description": "Specific suggestion",
                        "rationale": "Why this is recommended",
                        "priority": "high/medium/low",
                        "expected_impact": "Potential positive impact"
                    }}
                ]
            }}
            
            Consider:
            1. Portfolio diversification
            2. Risk concentration
            3. Sector allocation
            4. Market cycle positioning
            5. Hedging opportunities
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a portfolio manager with expertise in asset allocation and risk management. Provide practical, implementable suggestions."
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            suggestions = result.get('suggestions', [])
            
            self.logger.info(f"Generated {len(suggestions)} portfolio optimization suggestions")
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Error generating portfolio optimization suggestions: {str(e)}")
            return []
