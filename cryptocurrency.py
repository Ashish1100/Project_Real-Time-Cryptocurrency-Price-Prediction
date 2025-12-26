"""
Real-TimeCryptocurrency Price Prediction with LSTM

"""

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import warnings
import streamlit as st
from datetime import datetime, timedelta
import logging
from typing import Tuple, Optional
import os
import gc

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configuration
CONFIG = {
    'timestep': 60,
    'lstm_units': 64,
    'dropout_rate': 0.2,
    'epochs': 50,
    'batch_size': 16,
    'validation_split': 0.1,
    'early_stopping_patience': 10,
    'min_data_points': 150,
    'forecast_days_max': 30,
    'test_split': 0.2
}

@st.cache_data(ttl=3600)
def load_crypto_data(symbol: str, period: str = "1y") -> Optional[pd.DataFrame]:
    """Load and validate cryptocurrency data from Yahoo Finance."""
    try:
        with st.spinner(f"Loading data for {symbol}..."):
            data = yf.download(symbol, period=period, interval="1d", progress=False)
            
            if data.empty:
                st.error(f"No data found for {symbol}. Please check the symbol.")
                return None
            
            if isinstance(data, pd.DataFrame):
                data = data[['Close']].dropna()
            else:
                data = pd.DataFrame(data).dropna()
            
            if len(data) < CONFIG['min_data_points']:
                st.error(f"Insufficient data. Need at least {CONFIG['min_data_points']} data points. Got {len(data)}.")
                return None
            
            logger.info(f"Loaded {len(data)} data points for {symbol}")
            return data
    except Exception as e:
        logger.error(f"Data loading error: {str(e)}")
        st.error(f"Failed to load data: {str(e)}")
        return None

def create_scaler() -> MinMaxScaler:
    """Create a MinMaxScaler instance."""
    return MinMaxScaler(feature_range=(0, 1))

def prepare_sequences(data: np.ndarray, timestep: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences for LSTM training with proper reshaping."""
    X, y = [], []
    
    if data.ndim > 1:
        data = data.flatten()
    
    for i in range(len(data) - timestep):
        X.append(data[i:(i + timestep)])
        y.append(data[i + timestep])
    
    X = np.array(X).reshape(-1, timestep, 1)
    y = np.array(y)
    
    return X, y

def build_lstm_model(input_shape: Tuple[int, int]) -> Sequential:
    """Build and compile LSTM model architecture."""
    model = Sequential([
        LSTM(CONFIG['lstm_units'], return_sequences=True, input_shape=input_shape),
        Dropout(CONFIG['dropout_rate']),
        LSTM(CONFIG['lstm_units'], return_sequences=False),
        Dropout(CONFIG['dropout_rate']),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='huber', metrics=['mae'])
    return model

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate comprehensive performance metrics - returns Python floats."""
    try:
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100)
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'R² Score': r2,
            'MAPE': mape
        }
    except Exception as e:
        logger.error(f"Metrics calculation error: {str(e)}")
        return {'RMSE': 0.0, 'MAE': 0.0, 'R² Score': 0.0, 'MAPE': 0.0}

def generate_forecast(model: Sequential, last_sequence: np.ndarray, scaler: MinMaxScaler, days: int) -> np.ndarray:
    """Generate multi-day forecast with proper sequence management."""
    try:
        predictions = []
        current_sequence = last_sequence.copy()
        
        progress_placeholder = st.empty()
        
        for i in range(days):
            pred = model.predict(
                current_sequence.reshape(1, -1, 1),
                verbose=0,
                batch_size=1
            )
            
            pred_value = float(pred[0, 0])
            predictions.append(pred_value)
            
            current_sequence = np.append(current_sequence[1:], pred_value)
            
            progress = (i + 1) / days
            progress_placeholder.progress(progress, f"Forecasting: {i+1}/{days} days")
        
        progress_placeholder.empty()
        
        if not predictions:
            raise ValueError("No predictions generated")
        
        predictions_array = np.array(predictions).reshape(-1, 1)
        forecast = scaler.inverse_transform(predictions_array)
        
        return forecast.flatten()
    except Exception as e:
        logger.error(f"Forecast generation failed: {str(e)}")
        st.error(f"Forecast generation failed: {str(e)}")
        raise

def create_plotly_chart(
    data: pd.DataFrame,
    train_pred: np.ndarray,
    test_pred: np.ndarray,
    forecast: np.ndarray,
    forecast_days: int,
    symbol: str,
    train_size: int,
    test_size: int
) -> go.Figure:
    """Create interactive Plotly chart with professional styling - BALANCED LIGHT/DARK MODE."""
    try:
        # Create figure
        fig = go.Figure()
        
        # Convert data index to proper format
        actual_dates = pd.to_datetime(data.index)
        actual_prices = data['Close'].values.flatten()
        
        # Add actual price line
        fig.add_trace(
            go.Scatter(
                x=actual_dates,
                y=actual_prices,
                name='Actual Price',
                line=dict(color='#1f77b4', width=2),
                mode='lines',
                hovertemplate='<b>Actual Price</b><br>Date: %{x|%Y-%m-%d}<br>Price: $%{y:,.2f}<extra></extra>'
            )
        )
        
        # Add train predictions
        if len(train_pred) > 0:
            train_start_idx = CONFIG['timestep']
            train_end_idx = CONFIG['timestep'] + train_size
            
            if train_end_idx <= len(actual_dates):
                train_dates = actual_dates[train_start_idx:train_end_idx]
                train_prices = train_pred.flatten()
                
                if len(train_dates) == len(train_prices):
                    fig.add_trace(
                        go.Scatter(
                            x=train_dates,
                            y=train_prices,
                            name='Train Predictions',
                            line=dict(color='#2ca02c', width=2, dash='dot'),
                            mode='lines',
                            hovertemplate='<b>Train Pred</b><br>Date: %{x|%Y-%m-%d}<br>Price: $%{y:,.2f}<extra></extra>'
                        )
                    )
        
        # Add test predictions
        if len(test_pred) > 0:
            test_start_idx = CONFIG['timestep'] + train_size
            test_end_idx = test_start_idx + test_size
            
            if test_end_idx <= len(actual_dates):
                test_dates = actual_dates[test_start_idx:test_end_idx]
                test_prices = test_pred.flatten()
                
                if len(test_dates) == len(test_prices):
                    fig.add_trace(
                        go.Scatter(
                            x=test_dates,
                            y=test_prices,
                            name='Test Predictions',
                            line=dict(color='#ff7f0e', width=2, dash='dot'),
                            mode='lines',
                            hovertemplate='<b>Test Pred</b><br>Date: %{x|%Y-%m-%d}<br>Price: $%{y:,.2f}<extra></extra>'
                        )
                    )
        
        # Generate future dates for forecast
        last_date = actual_dates[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days, freq='D')
        forecast_prices = forecast.flatten()
        
        # Add confidence interval (upper bound)
        upper_bound = forecast_prices * 1.05
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=upper_bound,
                fill=None,
                line=dict(color='rgba(255,0,0,0)', width=0),
                showlegend=False,
                hoverinfo='skip',
                name='Upper Bound'
            )
        )
        
        # Add confidence interval (lower bound and fill)
        lower_bound = forecast_prices * 0.95
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=lower_bound,
                fill='tonexty',
                line=dict(color='rgba(255,0,0,0)', width=0),
                fillcolor='rgba(214, 39, 40, 0.2)',
                name='Confidence Interval (±5%)',
                hovertemplate='<b>Confidence Interval</b><br>Date: %{x|%Y-%m-%d}<br>Price: $%{y:,.2f}<extra></extra>'
            )
        )
        
        # Add forecast line
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=forecast_prices,
                name=f'{forecast_days}-Day Forecast',
                line=dict(color='#d62728', width=3),
                mode='lines+markers',
                marker=dict(size=6),
                hovertemplate='<b>Forecast</b><br>Date: %{x|%Y-%m-%d}<br>Price: $%{y:,.2f}<extra></extra>'
            )
        )
        
        # layout with balanced light/dark mode styling
        fig.update_layout(
            title=f'{symbol} LSTM Price Prediction Analysis',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            height=700,
            template='plotly',  # Use neutral template for better light/dark balance
            hovermode='x unified',
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1,
                bgcolor='rgba(255, 255, 255, 0.85)',  # Light white background for legend
                font=dict(size=11, color='rgba(0, 0, 0, 0.9)')  # Dark text for legend labels
            ),
            margin=dict(b=100, l=60, r=60, t=80),
            font=dict(size=11, color='rgba(0, 0, 0, 0.85)'),  # Dark text for all labels
            plot_bgcolor='rgba(240, 242, 246, 0.6)',  # Very light gray background
            paper_bgcolor='rgba(255, 255, 255, 1)',  # White paper background
            title_font=dict(size=16, color='rgba(0, 0, 0, 0.9)'),  # Dark title text
            xaxis=dict(
                title_font=dict(color='rgba(0, 0, 0, 0.85)', size=12),
                tickfont=dict(color='rgba(0, 0, 0, 0.75)', size=10),
                gridcolor='rgba(128, 128, 128, 0.15)',
                zeroline=False,
                showgrid=True
            ),
            yaxis=dict(
                title_font=dict(color='rgba(0, 0, 0, 0.85)', size=12),
                tickfont=dict(color='rgba(0, 0, 0, 0.75)', size=10),
                gridcolor='rgba(128, 128, 128, 0.15)',
                zeroline=False,
                showgrid=True
            )
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Chart creation error: {str(e)}")
        st.error(f"Chart creation failed: {str(e)}")
        raise

def main():
    """Main application function."""
    st.set_page_config(
        page_title="Crypto Predict Pro",
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state for persistence across reruns
    if 'forecast_complete' not in st.session_state:
        st.session_state.forecast_complete = False
        st.session_state.forecast_data = None
        st.session_state.metrics_data = None
    
    st.markdown("""
        <style>
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            color: white;
            text-align: center;
            font-size: 18px;
            font-weight: bold;
        }
        .disclaimer {
            background-color: #fff3cd;
            border: 1px solid #ffc107;
            padding: 15px;
            border-radius: 5px;
            margin-top: 20px;
            font-size: 12px;
            color: #856404;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("# Crypto Predict Pro")
    st.markdown("*Real-Time Cryptocurrency Price Forecasting*")
    
    # Add Pic2 image below title
    try:
        st.image("Pic2.png", use_container_width=True)
    except Exception as e:
        logger.warning(f"Pic2.png not found: {str(e)}")
    
    with st.sidebar:
        # Add Pic1 image at the top of sidebar (before Model Configuration)
        try:
            st.sidebar.image("Pic1.png", use_container_width=True)
        except Exception as e:
            logger.warning(f"Pic1.png not found: {str(e)}")
        
        st.header("Model Configuration")
        
        crypto_symbol = st.selectbox(
            "Select Cryptocurrency",
            options=["BTC-USD", "ETH-USD", "XRP-USD", "ADA-USD", "SOL-USD"],
            index=0,
            help="Choose the cryptocurrency pair to analyze"
        )
        
        data_period = st.select_slider(
            "Training Data Period",
            options=["3mo", "6mo", "1y", "2y", "3y", "4y", "5y"],
            value="1y",
            help="Historical data range for training"
        )
        
        forecast_days = st.slider(
            "Forecast Horizon (Days)",
            min_value=1,
            max_value=CONFIG['forecast_days_max'],
            value=7,
            help="Number of days to forecast into the future"
        )
        
        st.subheader("Advanced Parameters")
        lstm_units = st.slider(
            "LSTM Units",
            min_value=32,
            max_value=128,
            value=CONFIG['lstm_units'],
            step=16
        )
        
        dropout_rate = st.slider(
            "Dropout Rate",
            min_value=0.0,
            max_value=0.5,
            value=CONFIG['dropout_rate'],
            step=0.05
        )
        
        epochs = st.slider(
            "Training Epochs",
            min_value=10,
            max_value=100,
            value=CONFIG['epochs'],
            step=10
        )
        
        st.markdown("""
    <style>
        .stButton > button {
            background-color: #0099FF;
            color: white;
            border: none !important;         
            outline: none !important;        
        }
        .stButton > button:hover {
            background-color: #67C090;
            border: none !important;         
            outline: none !important;
        }
    </style>
""", unsafe_allow_html=True)


        
        generate_button = st.button(
            "Generate Forecast",
            use_container_width=True,
            type="primary"
        )
    
    if generate_button:
        # Reset state when generating new forecast
        st.session_state.forecast_complete = False
        
        CONFIG['lstm_units'] = lstm_units
        CONFIG['dropout_rate'] = dropout_rate
        CONFIG['epochs'] = epochs
        
        data = load_crypto_data(crypto_symbol, period=data_period)
        
        if data is not None:
            with st.spinner("Processing and training model..."):
                try:
                    st.info("Step 1: Normalizing data...")
                    scaler = create_scaler()
                    scaled_data = scaler.fit_transform(data[['Close']])
                    
                    st.info("Step 2: Preparing sequences...")
                    X, y = prepare_sequences(scaled_data, CONFIG['timestep'])
                    
                    st.info("Step 3: Splitting data into train/test...")
                    split_idx = int(len(X) * (1 - CONFIG['test_split']))
                    
                    X_train, X_test = X[:split_idx], X[split_idx:]
                    y_train, y_test = y[:split_idx], y[split_idx:]
                    
                    train_size = len(X_train)
                    test_size = len(X_test)
                    
                    st.success(f"Train set: {train_size} samples | Test set: {test_size} samples created!")
                    
                    st.info("Step 4: Building and training LSTM model...")
                    model = build_lstm_model((CONFIG['timestep'], 1))
                    
                    early_stop = EarlyStopping(
                        monitor='val_loss',
                        patience=CONFIG['early_stopping_patience'],
                        restore_best_weights=True
                    )
                    
                    history = model.fit(
                        X_train, y_train,
                        epochs=CONFIG['epochs'],
                        batch_size=CONFIG['batch_size'],
                        validation_split=CONFIG['validation_split'],
                        callbacks=[early_stop],
                        verbose=0
                    )
                    
                    st.success("Model training completed!")
                    
                    st.info("Step 5: Generating predictions...")
                    train_pred = model.predict(X_train, verbose=0)
                    test_pred = model.predict(X_test, verbose=0)
                    
                    train_pred_original = scaler.inverse_transform(train_pred)
                    test_pred_original = scaler.inverse_transform(test_pred)
                    
                    train_metrics = calculate_metrics(
                        scaler.inverse_transform(y_train.reshape(-1, 1)),
                        train_pred_original
                    )
                    
                    test_metrics = calculate_metrics(
                        scaler.inverse_transform(y_test.reshape(-1, 1)),
                        test_pred_original
                    )
                    
                    st.info("Step 6: Forecasting future prices...")
                    last_sequence = scaled_data[-CONFIG['timestep']:]
                    forecast = generate_forecast(model, last_sequence, scaler, forecast_days)
                    
                    st.success("Forecast generated successfully!")
                    
                    # Create all data before storing in session state
                    forecast_dates = pd.date_range(start=data.index[-1], periods=forecast_days + 1, freq='D')[1:]
                    
                    forecast_df = pd.DataFrame({
                        'Date': forecast_dates.strftime('%Y-%m-%d'),
                        'Forecast Price': [float(x) for x in forecast],
                        'Upper Bound': [float(x) for x in forecast * 1.05],
                        'Lower Bound': [float(x) for x in forecast * 0.95]
                    })
                    
                    fig = create_plotly_chart(
                        data,
                        train_pred_original.flatten(),
                        test_pred_original.flatten(),
                        forecast,
                        forecast_days,
                        crypto_symbol,
                        train_size,
                        test_size
                    )
                    
                    # Store all results in session state
                    st.session_state.forecast_complete = True
                    st.session_state.metrics_data = {
                        'latest_price': float(data['Close'].iloc[-1]),
                        'predicted_price': float(forecast[-1]),
                        'price_change': float(((float(forecast[-1]) - float(data['Close'].iloc[-1])) / float(data['Close'].iloc[-1])) * 100),
                        'train_metrics': train_metrics,
                        'test_metrics': test_metrics,
                        'forecast_df': forecast_df,
                        'fig': fig,
                        'forecast_days': forecast_days,
                        'forecast': forecast
                    }
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    logger.error(f"Main execution error: {str(e)}")
    
    # Display results from session state (persists after download!)
    if st.session_state.forecast_complete and st.session_state.metrics_data:
        metrics = st.session_state.metrics_data
        
        # Display metrics
        st.header("Model Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Test RMSE",
                f"${metrics['test_metrics']['RMSE']:.2f}",
                delta=f"Train: ${metrics['train_metrics']['RMSE']:.2f}"
            )
        
        with col2:
            st.metric(
                "Test MAE",
                f"${metrics['test_metrics']['MAE']:.2f}",
                delta=f"Train: ${metrics['train_metrics']['MAE']:.2f}"
            )
        
        with col3:
            st.metric(
                "Test R² Score",
                f"{metrics['test_metrics']['R² Score']:.4f}",
                delta=f"Train: {metrics['train_metrics']['R² Score']:.4f}"
            )
        
        with col4:
            st.metric(
                "Test MAPE",
                f"{metrics['test_metrics']['MAPE']:.2f}%",
                delta=f"Train: {metrics['train_metrics']['MAPE']:.2f}%"
            )
        
        # Price predictions
        st.header("Prediction Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Current Price",
                f"${metrics['latest_price']:,.2f}"
            )
        
        with col2:
            st.metric(
                f"Predicted Price ({metrics['forecast_days']}-Day)",
                f"${metrics['predicted_price']:,.2f}",
                delta=f"{metrics['price_change']:+.2f}%"
            )
        
        with col3:
            min_forecast = float(metrics['forecast'].min())
            max_forecast = float(metrics['forecast'].max())
            st.metric(
                "Forecast Range",
                f"${min_forecast:,.2f} - ${max_forecast:,.2f}"
            )
        
        # Interactive chart
        st.header("Price Chart")
        st.plotly_chart(metrics['fig'], use_container_width=True)
        
        # Detailed forecast table
        st.header("Detailed Forecast")
        st.dataframe(metrics['forecast_df'], use_container_width=True)
        
        # Download forecast
        st.header("Export Data")
        csv = metrics['forecast_df'].to_csv(index=False)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.download_button(
                label="📥 Download Forecast as CSV",
                data=csv,
                file_name=f"BTC-USD_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        
        with col2:
            if st.button("New Forecast", use_container_width=True):
                st.session_state.forecast_complete = False
                st.rerun()
        
        # Disclaimer
        st.markdown("""
            <div class="disclaimer">
            <strong>⚠️ Important Disclaimer:</strong><br>
            This prediction is not financial advice. Cryptocurrency markets are highly volatile and unpredictable. 
            Past performance does not guarantee future results. Always conduct your own research and consult with financial 
            advisors before making investment decisions.
            </div>
        """, unsafe_allow_html=True)
        
        # Footer
        st.markdown("---")
        st.markdown(
            "**Crypto Predict Pro** | Created with ❤️ by Ashish Saha using TensorFlow, Keras & Streamlit | "
            "© 2025 Ashish Saha"
        )
        
        gc.collect()

if __name__ == "__main__":
    main()
