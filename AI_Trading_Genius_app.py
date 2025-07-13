import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import threading
import time
import os
import numpy as np

from trading_system.data_processor import DataProcessor
from trading_system.ml_models import MLModelManager
from trading_system.trading_strategies import StrategyManager
from trading_system.risk_manager import RiskManager
from trading_system.portfolio_manager import PortfolioManager
from trading_system.ai_advisor import AIAdvisor
from trading_system.backtester import Backtester
from trading_system.trade_executor import TradeExecutor
from utils.logger import TradingLogger
from utils.config import TradingConfig

# Initialize session state
if 'trading_system_initialized' not in st.session_state:
    st.session_state.trading_system_initialized = False
    st.session_state.system_running = False
    st.session_state.paper_trading = True
    st.session_state.emergency_stop = False

# Initialize components
@st.cache_resource
def initialize_trading_system():
    """Initialize all trading system components"""
    try:
        config = TradingConfig()
        logger = TradingLogger()
        
        data_processor = DataProcessor(config, logger)
        ml_models = MLModelManager(config, logger)
        risk_manager = RiskManager(config, logger)
        portfolio_manager = PortfolioManager(config, logger)
        ai_advisor = AIAdvisor(config, logger)
        strategy_manager = StrategyManager(config, logger, ml_models, ai_advisor)
        backtester = Backtester(config, logger, data_processor)
        trade_executor = TradeExecutor(config, logger, risk_manager, portfolio_manager)
        
        return {
            'config': config,
            'logger': logger,
            'data_processor': data_processor,
            'ml_models': ml_models,
            'strategy_manager': strategy_manager,
            'risk_manager': risk_manager,
            'portfolio_manager': portfolio_manager,
            'ai_advisor': ai_advisor,
            'backtester': backtester,
            'trade_executor': trade_executor
        }
    except Exception as e:
        st.error(f"Failed to initialize trading system: {str(e)}")
        return None

def main():
    st.set_page_config(
        page_title="AI Trading System - India",
        page_icon="🇮🇳",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header with warnings
    st.title("🇮🇳 AI Trading System - Indian Stock Market")
    st.caption("NSE/BSE Automated Trading • All amounts in Indian Rupees (₹)")
    
    # Critical warnings
    st.error("⚠️ CRITICAL DISCLAIMER: This system is for educational purposes only. Trading involves substantial risk of loss. Only risk capital you can afford to lose completely.")
    
    with st.expander("📋 Important Risk Warnings", expanded=False):
        st.warning("""
        **IMPORTANT RISK WARNINGS:**
        - No trading system can guarantee profits or eliminate losses
        - Markets are inherently unpredictable and all trading involves substantial risk
        - Past performance does not guarantee future results
        - This system is experimental and for educational purposes
        - Always start with paper trading before using real money
        - Never risk more than you can afford to lose completely
        """)
    
    # Initialize system
    components = initialize_trading_system()
    if not components:
        st.stop()
    
    # Sidebar controls
    st.sidebar.header("🎛️ System Controls")
    
    # Trading mode
    st.session_state.paper_trading = st.sidebar.checkbox(
        "Paper Trading Mode", 
        value=st.session_state.paper_trading,
        help="Recommended: Start with paper trading to test strategies"
    )
    
    if not st.session_state.paper_trading:
        st.sidebar.warning("⚠️ REAL MONEY TRADING ENABLED")
        real_trading_confirm = st.sidebar.checkbox("I understand the risks of real money trading")
        if not real_trading_confirm:
            st.sidebar.error("Please confirm you understand the risks")
            st.stop()
    
    # Emergency stop
    if st.sidebar.button("🛑 EMERGENCY STOP", type="primary"):
        st.session_state.emergency_stop = True
        st.session_state.system_running = False
        components['trade_executor'].emergency_stop()
        st.sidebar.success("Emergency stop activated!")
    
    # System status
    st.sidebar.header("📊 System Status")
    status_container = st.sidebar.container()
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Dashboard", 
        "🧠 AI Strategies", 
        "📊 Performance", 
        "⚖️ Risk Management", 
        "🔄 Backtesting", 
        "⚙️ Settings"
    ])
    
    with tab1:
        render_dashboard(components)
    
    with tab2:
        render_ai_strategies(components)
    
    with tab3:
        render_performance(components)
    
    with tab4:
        render_risk_management(components)
    
    with tab5:
        render_backtesting(components)
    
    with tab6:
        render_settings(components)
    
    # Update status
    with status_container:
        if st.session_state.system_running:
            st.success("🟢 System Running")
        else:
            st.info("🔵 System Stopped")
        
        if st.session_state.paper_trading:
            st.info("📝 Paper Trading")
        else:
            st.warning("💰 Live Trading")
        
        if st.session_state.emergency_stop:
            st.error("🛑 Emergency Stop")

def render_dashboard(components):
    """Render main dashboard"""
    st.header("📈 Trading Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        # Get portfolio data
        portfolio_data = components['portfolio_manager'].get_portfolio_summary()
        
        # Get USD to INR conversion rate
        usd_inr_rate = components['data_processor'].get_usd_inr_rate()
        
        with col1:
            portfolio_value_usd = portfolio_data.get('total_value', 0)
            portfolio_value_inr = components['data_processor'].convert_usd_to_inr(portfolio_value_usd)
            st.metric(
                "Portfolio Value", 
                f"₹{portfolio_value_inr:,.2f}",
                f"{portfolio_data.get('daily_pnl', 0):+.2f}"
            )
        
        with col2:
            daily_pnl_usd = portfolio_data.get('daily_pnl', 0)
            daily_pnl_inr = components['data_processor'].convert_usd_to_inr(daily_pnl_usd)
            st.metric(
                "Daily P&L", 
                f"₹{daily_pnl_inr:,.2f}",
                f"{portfolio_data.get('daily_pnl_pct', 0):+.2f}%"
            )
        
        with col3:
            st.metric(
                "Open Positions", 
                portfolio_data.get('open_positions', 0)
            )
        
        with col4:
            st.metric(
                "Risk Score", 
                f"{components['risk_manager'].get_portfolio_risk_score():.2f}"
            )
        
        # Exchange rate info (for reference only)
        st.subheader("💱 Exchange Rate")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "USD/INR Rate", 
                f"₹{usd_inr_rate:.4f}",
                help="Live exchange rate - All amounts displayed in INR"
            )
        
        with col2:
            st.info("💰 All trading amounts shown in Indian Rupees (₹)")
        
        # Control buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("▶️ Start Trading System", disabled=st.session_state.system_running):
                start_trading_system(components)
        
        with col2:
            if st.button("⏸️ Stop Trading System", disabled=not st.session_state.system_running):
                stop_trading_system(components)
        
        with col3:
            if st.button("🔄 Refresh Data"):
                st.rerun()
        
        # Real-time market data
        st.subheader("📊 Market Overview")
        market_data = components['data_processor'].get_market_overview()
        
        if market_data is not None and not market_data.empty:
            fig = px.line(
                market_data, 
                x='time', 
                y='price', 
                color='symbol',
                title="Real-time Market Data"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Market data will appear here when the system is running")
        
        # Current positions with USD/INR conversion
        st.subheader("📊 Current Positions")
        positions = components['portfolio_manager'].get_all_positions()
        
        if positions:
            for symbol, position in positions.items():
                with st.expander(f"📈 {symbol} - {position['quantity']} shares"):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    market_value_usd = position.get('market_value', 0)
                    market_value_inr = components['data_processor'].convert_usd_to_inr(market_value_usd)
                    
                    pnl_usd = position.get('unrealized_pnl', 0)
                    pnl_inr = components['data_processor'].convert_usd_to_inr(pnl_usd)
                    
                    entry_price_usd = position.get('entry_price', 0)
                    entry_price_inr = components['data_processor'].convert_usd_to_inr(entry_price_usd)
                    
                    current_price_usd = position.get('current_price', 0)
                    current_price_inr = components['data_processor'].convert_usd_to_inr(current_price_usd)
                    
                    with col1:
                        st.metric("Market Value", f"₹{market_value_inr:,.2f}")
                    
                    with col2:
                        st.metric("P&L", f"₹{pnl_inr:,.2f}")
                    
                    with col3:
                        st.metric("Entry Price", f"₹{entry_price_inr:,.2f}")
                    
                    with col4:
                        st.metric("Current Price", f"₹{current_price_inr:,.2f}")
        else:
            st.info("No open positions")
        
        # Active strategies
        st.subheader("🎯 Active Strategies")
        strategies = components['strategy_manager'].get_active_strategies()
        
        if strategies:
            for strategy in strategies:
                with st.expander(f"📈 {strategy['name']} - {strategy['status']}"):
                    col1, col2, col3 = st.columns(3)
                    
                    pnl_usd = strategy.get('pnl', 0)
                    pnl_inr = components['data_processor'].convert_usd_to_inr(pnl_usd)
                    
                    with col1:
                        st.metric("P&L", f"₹{pnl_inr:,.2f}")
                    with col2:
                        st.metric("Win Rate", f"{strategy.get('win_rate', 0):.1f}%")
                    with col3:
                        st.metric("Trades", strategy.get('total_trades', 0))
        else:
            st.info("No active strategies running")
        
    except Exception as e:
        st.error(f"Error rendering dashboard: {str(e)}")

def render_ai_strategies(components):
    """Render AI strategies tab"""
    st.header("🧠 AI-Powered Trading Strategies")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🤖 AI Market Analysis")
        
        if st.button("🔍 Generate AI Market Insights"):
            with st.spinner("AI analyzing market conditions..."):
                try:
                    insights = components['ai_advisor'].get_market_insights()
                    st.success("AI Analysis Complete!")
                    
                    for insight in insights:
                        st.write(f"**{insight['type']}:** {insight['content']}")
                        if insight.get('confidence'):
                            st.progress(insight['confidence'])
                except Exception as e:
                    st.error(f"AI analysis failed: {str(e)}")
        
        st.subheader("📈 Strategy Performance")
        strategy_performance = components['strategy_manager'].get_strategy_performance()
        
        if strategy_performance:
            df = pd.DataFrame(strategy_performance)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("Strategy performance data will appear after trading begins")
    
    with col2:
        st.subheader("⚙️ Strategy Configuration")
        
        # Strategy selection
        available_strategies = [
            "Neural Network Momentum",
            "Ensemble Trend Following",
            "AI Pattern Recognition",
            "Adaptive Mean Reversion",
            "Multi-Factor ML Model"
        ]
        
        selected_strategies = st.multiselect(
            "Select Active Strategies",
            available_strategies,
            default=available_strategies[:2]
        )
        
        # Risk parameters
        st.slider("Max Position Size (%)", 1, 20, 5)
        st.slider("Stop Loss (%)", 1, 10, 3)
        st.slider("Take Profit (%)", 2, 20, 8)
        
        if st.button("💾 Update Strategy Config"):
            st.success("Strategy configuration updated!")

def render_performance(components):
    """Render performance analysis tab"""
    st.header("📊 Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Portfolio Performance")
        
        # Generate sample performance data
        try:
            performance_data = components['portfolio_manager'].get_performance_history()
            
            if performance_data is not None and not performance_data.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=performance_data['date'],
                    y=performance_data['portfolio_value'],
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='blue', width=2)
                ))
                
                fig.update_layout(
                    title="Portfolio Value Over Time",
                    xaxis_title="Date",
                    yaxis_title="Value ($)",
                    hovermode='x'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Performance data will be available after trading begins")
        except Exception as e:
            st.error(f"Error loading performance data: {str(e)}")
    
    with col2:
        st.subheader("📊 Risk Metrics")
        
        try:
            risk_metrics = components['risk_manager'].get_risk_metrics()
            
            for metric, value in risk_metrics.items():
                if isinstance(value, (int, float)):
                    st.metric(metric.replace('_', ' ').title(), f"{value:.4f}")
        except Exception as e:
            st.error(f"Error loading risk metrics: {str(e)}")

def render_risk_management(components):
    """Render risk management tab"""
    st.header("⚖️ Risk Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🛡️ Risk Controls")
        
        max_portfolio_risk = st.slider(
            "Maximum Portfolio Risk (%)",
            min_value=1.0,
            max_value=10.0,
            value=5.0,
            step=0.5
        )
        
        max_position_size = st.slider(
            "Maximum Position Size (%)",
            min_value=1.0,
            max_value=25.0,
            value=10.0,
            step=0.5
        )
        
        stop_loss_threshold = st.slider(
            "Global Stop Loss (%)",
            min_value=1.0,
            max_value=15.0,
            value=5.0,
            step=0.5
        )
        
        if st.button("💾 Update Risk Parameters"):
            components['risk_manager'].update_risk_parameters({
                'max_portfolio_risk': max_portfolio_risk / 100,
                'max_position_size': max_position_size / 100,
                'stop_loss_threshold': stop_loss_threshold / 100
            })
            st.success("Risk parameters updated!")
    
    with col2:
        st.subheader("📊 Current Risk Status")
        
        try:
            risk_status = components['risk_manager'].get_risk_status()
            
            for risk_type, details in risk_status.items():
                status = details.get('status', 'unknown')
                value = details.get('value', 0)
                
                if status == 'safe':
                    st.success(f"✅ {risk_type}: {value:.2%}")
                elif status == 'warning':
                    st.warning(f"⚠️ {risk_type}: {value:.2%}")
                else:
                    st.error(f"🚨 {risk_type}: {value:.2%}")
        except Exception as e:
            st.error(f"Error loading risk status: {str(e)}")

def render_backtesting(components):
    """Render backtesting tab"""
    st.header("🔄 Strategy Backtesting")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⚙️ Backtest Configuration")
        
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=365)
        )
        
        end_date = st.date_input(
            "End Date",
            value=datetime.now()
        )
        
        initial_capital = st.number_input(
            "Initial Capital ($)",
            min_value=1000,
            max_value=1000000,
            value=100000
        )
        
        strategy_to_test = st.selectbox(
            "Strategy to Test",
            ["Neural Network Momentum", "Ensemble Trend Following", "AI Pattern Recognition"]
        )
        
        if st.button("🚀 Run Backtest"):
            with st.spinner("Running backtest..."):
                try:
                    results = components['backtester'].run_backtest(
                        strategy=strategy_to_test,
                        start_date=start_date,
                        end_date=end_date,
                        initial_capital=initial_capital
                    )
                    
                    st.session_state.backtest_results = results
                    st.success("Backtest completed!")
                except Exception as e:
                    st.error(f"Backtest failed: {str(e)}")
    
    with col2:
        st.subheader("📊 Backtest Results")
        
        if hasattr(st.session_state, 'backtest_results'):
            results = st.session_state.backtest_results
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Return", f"{results.get('total_return', 0):.2%}")
            with col2:
                st.metric("Sharpe Ratio", f"{results.get('sharpe_ratio', 0):.2f}")
            with col3:
                st.metric("Max Drawdown", f"{results.get('max_drawdown', 0):.2%}")
            
            # Performance chart
            if 'equity_curve' in results:
                fig = px.line(
                    results['equity_curve'], 
                    x='date', 
                    y='equity',
                    title="Backtest Equity Curve"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run a backtest to see results here")

def render_settings(components):
    """Render settings tab"""
    st.header("⚙️ System Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔧 Trading Parameters")
        
        trading_frequency = st.selectbox(
            "Trading Frequency",
            ["1 minute", "5 minutes", "15 minutes", "1 hour", "Daily"]
        )
        
        max_trades_per_day = st.number_input(
            "Max Trades Per Day",
            min_value=1,
            max_value=1000,
            value=10
        )
        
        commission_rate = st.number_input(
            "Commission Rate (%)",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.01
        )
    
    with col2:
        st.subheader("🔑 API Configuration")
        
        openai_key_status = "✅ Connected" if os.getenv("OPENAI_API_KEY") else "❌ Not Found"
        st.info(f"OpenAI API: {openai_key_status}")
        
        st.subheader("📊 Data Sources")
        st.info("Yahoo Finance: ✅ Active")
        
        st.subheader("🔄 System Actions")
        
        if st.button("🧹 Clear Cache"):
            st.cache_resource.clear()
            st.success("Cache cleared!")
        
        if st.button("📁 Export Logs"):
            st.info("Log export functionality would be implemented here")

def start_trading_system(components):
    """Start the automated trading system"""
    try:
        st.session_state.system_running = True
        st.session_state.emergency_stop = False
        
        # Start trading thread
        def trading_loop():
            while st.session_state.system_running and not st.session_state.emergency_stop:
                try:
                    # Update market data
                    components['data_processor'].update_market_data()
                    
                    # Generate trading signals
                    signals = components['strategy_manager'].generate_signals()
                    
                    # Execute trades (only if not paper trading and signals exist)
                    if signals and not st.session_state.paper_trading:
                        components['trade_executor'].execute_trades(signals)
                    
                    # Update portfolio
                    components['portfolio_manager'].update_positions()
                    
                    time.sleep(60)  # Wait 1 minute between cycles
                    
                except Exception as e:
                    components['logger'].error(f"Trading loop error: {str(e)}")
                    time.sleep(10)
        
        # Start trading thread
        trading_thread = threading.Thread(target=trading_loop, daemon=True)
        trading_thread.start()
        
        st.success("✅ Trading system started!")
        
    except Exception as e:
        st.error(f"Failed to start trading system: {str(e)}")
        st.session_state.system_running = False

def stop_trading_system(components):
    """Stop the automated trading system"""
    try:
        st.session_state.system_running = False
        components['trade_executor'].stop_trading()
        st.success("⏸️ Trading system stopped!")
    except Exception as e:
        st.error(f"Error stopping trading system: {str(e)}")

if __name__ == "__main__":
    main()
