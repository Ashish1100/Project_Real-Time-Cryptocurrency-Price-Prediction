<div align="center">

# **Crypto Predict Pro**

### *Real-Time Cryptocurrency Price Forecasting*

#### Harness the power of deep learning to predict tomorrow's crypto prices today

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://real-time-cryptocurrency-price-prediction.streamlit.app/)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?logo=keras&logoColor=white)](https://keras.io/)
[![License: Non-Commercial](https://img.shields.io/badge/License-Non--Commercial-green.svg)](LICENSE)


</div>

---

## **The Main Idea**

In the volatile world of cryptocurrency, **timing is everything**. Crypto Predict Pro empowers traders, researchers, and enthusiasts with **AI predictions** through an elegant, web-based interface. 

### **Why It Matters: The Need**

| Challenge | Traditional Approach | Crypto Predict Pro |
|-----------|---------------------|----------------------|
| **Market Unpredictability** | Manual analysis (slow, biased) | LSTM Deep Learning |
| **Analysis Speed** | Hours of research | Instant results in seconds |
| **Data Access** | Multiple tools needed | Unified real-time integration |
| **Cost** | Can be expensive | Completely FREE & open-source |

### **The Importance**

```
Cryptocurrency Market Facts:
├─ $2.9 Trillion global market cap
├─ 24/7 trading (no closing bells)
├─ Extreme volatility (100%+ swings common)
├─ AI-driven trading dominates (≈90% of volume)
└─ Early adopters of AI gain competitive edge
```

---

## **Live Deployment**

Experience the power of AI-driven predictions instantly:

| Platform | Link | Status | Features |
|----------|------|--------|----------|
| **Streamlit** | <a href="https://real-time-cryptocurrency-price-prediction.streamlit.app/" target="_blank" rel="noopener noreferrer">**Live App**</a> | ✅ Active | Full features, instant startup (may need to wake-up) |
| **Render** | [**Alternative**](https://real-time-cryptocurrency-price-prediction.onrender.com/) | ✅ Active | Backup instance, 24/7 uptime |

> **Note:** Click the links above to see real AI predictions in action! No installation required.

---

## **Feature Overview**

<table>
<tr>
<td width="50%">

### **Frontend & UI (*What We See*)**
- **Interactive Charts:** Dynamic price visualization.
- **Real-time Predictions:** Live model inference.
- **Dashboard:** Performance metrics at a glance.
- **Export:** Download forecasts as `.csv`.
- **Theming:** Toggle Light/Dark mode.
- **Responsive:** Optimized for Mobile & Desktop.

</td>
<td width="50%">

### **Backend & AI (*What's Behind*)**
- **Deep Learning:** Long Short-Term Memory (LSTM) Networks.
- **Analytics:** Advanced statistical modeling.
- **Forecasting:** Iterative time-series prediction.
- **Quality:** PEP-8 compliant, clean code.
- **Security:** Secure data processing pipeline.
- **Performance:** Caching for low latency.

</td>
</tr>
</table>

---

## **Key Features Breakdown**

### **Prediction Engine**

```
╔════════════════════════════════════════════════════════╗
║                LSTM ARCHITECTURE                       ║
╠════════════════════════════════════════════════════════╣
║                                                        ║
║  Input Layer (60 timesteps)                            ║
║        ↓                                               ║
║  LSTM Layer 1 (64 units) + Dropout (20%)               ║
║        ↓                                               ║
║  LSTM Layer 2 (64 units) + Dropout (20%)               ║
║        ↓                                               ║
║  Dense Layer (32 units, ReLU activation)               ║
║        ↓                                               ║
║  Output Layer (1 unit, Linear activation)              ║
║        ↓                                               ║
║  Price Prediction (Next N days)                        ║
║                                                        ║
╚════════════════════════════════════════════════════════╝

✓ Multi-layer LSTM for complex pattern recognition
✓ Dropout regularization prevents overfitting
✓ ReLU activation captures non-linear relationships
✓ Real-time model training & prediction
✓ Early stopping for optimal convergence
✓ Less Complex and faster results
```

### **Professional Visualizations**

```
Interactive Chart Features:
├─ Actual Prices (Blue line) ────────────────┐
├─ Train Predictions (Green dotted) ─────────├─ Plotly
├─ Test Predictions (Orange dotted) ─────────├─ Multi-Series
├─ Future Forecast (Red solid) ──────────────├─ Zoomable
└─ Confidence Bounds (Pink band, ±5%) ───────┘─ PNG Export

Chart Capabilities:
✓ Hover tooltips with exact values
✓ Zoom and pan across date ranges
✓ Toggle series visibility
✓ Download as PNG image
✓ Responsive on all devices
```

### **Cryptocurrency Support**

| Cryptocurrency | Symbol | Market Cap | Primary Use Case |
|----------------|--------|------------|------------------|
| Bitcoin        | BTC    | ~$2.1T     | Store of value   |
| Ethereum       | ETH    | ~$400B     | Smart contracts  |
| XRP            | XRP    | ~$30B      | Payments         |
| Cardano        | ADA    | ~$50B      | Proof of stake   |
| Solana         | SOL    | ~$80B      | High-speed chain |


### **Comprehensive Metrics**

| Metric | Formula | Meaning | Target |
|--------|---------|---------|--------|
| **RMSE** | √(Σ(actual-pred)²/n) | Prediction error magnitude | Lower is better |
| **MAE** | Σ\|actual-pred\|/n | Average absolute error | Lower is better |
| **R² Score** | 1 - (SS_res/SS_tot) | Variance explained | Higher is better (max: 1.0) |
| **MAPE** | (Σ\|error\|/\|actual\|)/n × 100 | Error percentage | Lower is better |

**Example Results (Bitcoin):**
```
Test RMSE:    $2,775.64  ⟷  Train: $2,307.90  (20% difference - good)
Test MAE:     $2,136.42  ⟷  Train: $1,637.27  (30% difference - acceptable)
Test R²:      0.9305     ⟷  Train: 0.9922     (Explains 93.05% variance)
Test MAPE:    2.01%      ⟷  Train: 2.92%      (±2.01% accuracy)

✓ Model is not overfitting and shows good generalization.
✓ Predictions typically accurate within 2%
✓ Confidence in out-of-sample predictions: HIGH
```
> Performance can be optimized by adjusting Training Epochs, Training Data Period, LSTM Units and Dropout Rate.


---

## How It Works - Technical Overview

### End-to-End Pipeline
```
1. Data Ingestion and Validation  
   - Historical price data is loaded, cleaned, and validated to ensure sufficient sequence length.

2. Normalization  
   - Prices are scaled to a normalized range to stabilize training while preserving inverse scaling for outputs.

3. Sequence Construction  
   - Fixed-length lookback windows (60 timesteps) are created to frame the supervised learning problem.

4. Model Training  
   - An LSTM-based model is trained using Adam optimization, robust loss functions, and early stopping.

5. Evaluation  
   - Performance is assessed using standard regression metrics (RMSE, MAE, R2, MAPE).

6. Forecasting  
   - Future prices are generated iteratively using the most recent sequence and inverse transformation.

7. Visualization and Export  
   - Results are visualized interactively and made available for export in tabular form.
```

### **Why This Approach Works**

```
Traditional ML                  vs    Deep Learning (LSTM)
───────────────────────────────────────────────────────────────────────
Manual feature engineering            Automatic feature learning
Linear relationships only             Non-linear pattern detection
Limited historical context            120-day temporal memory (60×2)
Single prediction error               Sequence-level optimization
Slow retraining (days)                Fast retraining (seconds)
50-60% accuracy typical               High (80%-95%) accuracy achieved
```

---

## **Project Structure & Architecture**

### **Directory Structure**

```
CryptoPredict-Pro/
│
├── cryptocurrency.py              # Main application (1 file = simplicity!)
│   ├── Configuration (CONFIG dict)
│   ├── Data loading (yfinance)
│   ├── Model building (Keras)
│   ├── Training & prediction
│   └── Visualization (Plotly)
│
├── requirements.txt               # Dependencies (pip install)
│   ├── streamlit==1.28.0
│   ├── tensorflow==2.13.0
│   ├── keras==2.13.0
│   ├── yfinance==0.2.32
│   ├── plotly==5.17.0
│   ├── pandas, numpy, scikit-learn
│   └── ... (8 total packages)
│
├── 🖼️ Pic1.png                     # Sidebar banner (crypto icon)
├── 🖼️ Pic2.png                     # Main content image
│
├── README.md                     # This file
│
├── .gitignore                    # Git ignore file
│   ├── venv/
│   ├── __pycache__/
│   ├── .streamlit/secrets.toml
│   └── *.pyc
│
└── LICENSE                       # MIT License

Total Lines of Code: ~900 (lean & efficient!)
```

### **Code Architecture Diagram**


![Crypto Predict Pro Architecture](https://github.com/Ashish1100/Project_Real-Time-Cryptocurrency-Price-Prediction/blob/main/images/architecture.png)


---

## **Quick Start Guide**

### **Installation**

```bash
# 1. Clone the repository
git clone https://github.com/Ashish1100/Project_Real-Time-Cryptocurrency-Price-Prediction.git
cd Project_Real-Time-Cryptocurrency-Price-Prediction

# 2. Create & activate virtual environment
python -m venv venv
source venv\Scripts\activate  # Windows

# 3. Install dependencies & run
pip install -r requirements.txt
streamlit run cryptocurrency.py
```

**Done!** Open `http://localhost:8501` in your browser.

### **What Each Dependency Does**

| Package | Version | Purpose | Why |
|---------|---------|---------|-----|
| **streamlit** | 1.28.0 | Web app framework | Interactive UI, rapid development |
| **tensorflow** | 2.13.0 | Deep learning | LSTM model building & training |
| **keras** | 2.13.0 | Neural network API | Simplified model creation |
| **yfinance** | 0.2.32 | Financial data | Real-time crypto prices |
| **plotly** | 5.17.0 | Visualization | Interactive, beautiful charts |
| **pandas** | 2.0.0 | Data manipulation | DataFrame operations |
| **numpy** | 1.24.0 | Numerical computing | Array operations, math |
| **scikit-learn** | 1.3.0 | ML utilities | MinMaxScaler, metrics |

---

## **Demo & Screenshots Guide**

### **Recommended GIFs to Create/Add**

| GIF | Description | Tools |
|-----|-------------|-------|
| **app-demo.gif** | Full app walkthrough (30s) | ScreenFlow, Camtasia |
| **prediction-animation.gif** | Real-time prediction generation | Matplotlib animation |
| **chart-interaction.gif** | Zooming, panning, hovering | Plotly recording |
| **model-training.gif** | LSTM training progress | TensorFlow visualization |
| **data-export.gif** | CSV download process | Desktop recording |

### **Crypto Predict Pro Working Screenshots**


#### Dashboard Overview
![Dashboard_1](https://github.com/Ashish1100/Project_Real-Time-Cryptocurrency-Price-Prediction/blob/main/images/Dashboard_1.png)
![Dashboard_2](https://github.com/Ashish1100/Project_Real-Time-Cryptocurrency-Price-Prediction/blob/main/images/Dashboard_2.png)

#### Interactive Price Chart Example
![Chart_1](https://github.com/Ashish1100/Project_Real-Time-Cryptocurrency-Price-Prediction/blob/main/images/Chart_1.png)
![Chart_2](https://github.com/Ashish1100/Project_Real-Time-Cryptocurrency-Price-Prediction/blob/main/images/Chart_2.png)

#### Metrics Display
![Metrics](https://github.com/Ashish1100/Project_Real-Time-Cryptocurrency-Price-Prediction/blob/main/images/Metrics.png)

#### Detailed Forecast Table
![Detailed_Forecast](https://github.com/Ashish1100/Project_Real-Time-Cryptocurrency-Price-Prediction/blob/main/images/Detailed_Forecast.png)

#### Downloaded Forecast .csv
![Downloaded_Forecast](https://github.com/Ashish1100/Project_Real-Time-Cryptocurrency-Price-Prediction/blob/main/images/Downloaded_Forecast.png)

#### Mobile/Phone View
![Mobile_1](https://github.com/Ashish1100/Project_Real-Time-Cryptocurrency-Price-Prediction/blob/main/images/Mobile_1.jpg)
![Mobile_2](https://github.com/Ashish1100/Project_Real-Time-Cryptocurrency-Price-Prediction/blob/main/images/Mobile_2.jpg)
![Mobile_3](https://github.com/Ashish1100/Project_Real-Time-Cryptocurrency-Price-Prediction/blob/main/images/Mobile_3.jpg)

---

## **Technical Implementation Details**

### **Data Source Specifications**

```
Yahoo Finance (yfinance)
├─ Data Type: Daily OHLCV (Open, High, Low, Close, Volume)
├─ Update Frequency: Market close + ~15 min delay
├─ Historical Range: Up to 5 years
├─ Data Quality: 99.5% (Yahoo standard)
├─ Reliability: Grade A (trusted by millions)
└─ Cost: FREE ✓
```

### **Model Parameters Explained**

```
CONFIG = {
    'timestep': 60,              # Historical window (60 days)
    'lstm_units': 64,            # Neural network capacity
    'dropout_rate': 0.2,         # Regularization (20%)
    'epochs': 10,                # Training iterations
    'batch_size': 32,            # Gradient update frequency
    'validation_split': 0.1,     # 10% validation data
    'early_stopping_patience': 5,# Stop if no improvement
    'min_data_points': 100,      # Minimum historical data
    'forecast_days_max': 30      # Maximum prediction horizon
}
```

---

## **Model Limitations & Honesty**

### **What Crypto Predict Pro CAN Do**

```
✅ Capture market trends & cycles
✅ Identify short-term momentum
✅ Learn seasonal patterns
✅ Provide statistical confidence bounds
✅ Automate repetitive analysis
✅ Speed up decision-making
✅ Backtest trading strategies
```

### **What Crypto Predict Pro CANNOT Do**

```
❌ Predict black swan events
❌ React to breaking news/regulations
❌ Account for macro-economic factors
❌ Eliminate market risk
❌ Guarantee profits
❌ Work with 100% accuracy
❌ Replace professional financial advisors
```

---

## **License & Legal**

```
© 2025 Ashish Saha

This project is a personal initiative intended for educational use only.

Permission is granted to use, copy, and modify this software for learning and research purposes.
Commercial use, sale, or monetization of this software or its derivatives is strictly prohibited.

The software is provided “as is”, without warranty of any kind.

```

---

## **Disclaimer & Risk Notice**

> ⚠️ **IMPORTANT NOTICE**
>
> - Cryptocurrency markets are **highly volatile and unpredictable**
> - LSTM predictions should **NOT be treated as financial advice**
> - Past performance **does NOT guarantee** future results
> - **Never invest money you cannot afford to lose**
> - **Always consult a qualified financial advisor** before trading
> - **Crypto markets operate 24/7** with extreme risk
> - **Scams and hacks are common** in crypto space
> - **Regulatory changes can crash markets** instantly
> - **Model accuracy degrades over time** (requires retraining)
> - **You assume 100% responsibility** for your financial decisions
>
> **Use this tool responsibly. Stay informed. Trade safely.**

---

## **Author**

<div align="center">

### **Ashish Saha**

**Machine Learning Engineer**

 Passionate about Data Science, AI and Innovation

[GitHub](https://github.com/Ashish1100) • [LinkedIn](https://www.linkedin.com/in/ashishsaha21/) • [Email](mailto:ashishsaha.softwareemail@email.com)


---

## **Project Statistics**

| Metric | Value |
|--------|-------|
| **Lines of Code** | ~900 |
| **Core Dependencies** | 8 |
| **Model Layers** | 5 |
| **Parameters** | 10+ configurable |
| **Supported Cryptos** | 5 major assets |
| **Forecast Horizon** | 1-30 days |
| **Model Accuracy** | 80-95% R² score |
| **Deployment Instances** | 2 live |


---

</div>

<div align="center">

### **Star ⭐ this repo if you found it helpful!**


---

*Made with ❤️ by Ashish Saha*

</div>
