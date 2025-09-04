import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import plotly.graph_objs as go
from datetime import datetime

# ------------------ SETUP ------------------
st.set_page_config(page_title="Stock Price Prediction", layout="wide")
st.title("📈 Stock Price Prediction (2010–Today)")
st.sidebar.header("Forecast Settings")

# User input for stock ticker
ticker = st.sidebar.text_input("Stock Ticker Symbol", value="AAPL").upper()

start_date = "2010-01-01"
end_date = datetime.today().strftime('%Y-%m-%d')

# ------------------ FORECAST SETTINGS ------------------
time_unit = st.sidebar.selectbox("Forecast Unit", ["Days", "Weeks", "Months", "Years"])
unit_count = st.sidebar.selectbox("Number of Units", list(range(1, 61)), index=11)

# Convert to business days
if time_unit == "Days":
    forecast_days = unit_count
elif time_unit == "Weeks":
    forecast_days = unit_count * 5
elif time_unit == "Months":
    forecast_days = unit_count * 21
elif time_unit == "Years":
    forecast_days = unit_count * 252

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data(ticker, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    df.reset_index(inplace=True)
    return df

df_full = load_data(ticker, end_date)

if df_full.empty:
    st.error("❌ No data was loaded. Please try again later or check the ticker symbol.")
    st.stop()

# Prepare data for models
df_model = df_full[['Date', 'Close']].copy()
df_model.columns = ['ds', 'y']
latest_date = df_model['ds'].max()
st.success(f"✅ Latest available data: {latest_date.date()}")

# ------------------ CALCULATE MOVING AVERAGES ------------------
df_model['SMA_100'] = df_model['y'].rolling(window=100).mean()
df_model['SMA_200'] = df_model['y'].rolling(window=200).mean()
df_model['EMA'] = df_model['y'].ewm(span=50, adjust=False).mean()

# ------------------ MODEL FUNCTIONS ------------------
def arima_model(data, forecast_days):
    ts = data.set_index('ds')['y']
    model = ARIMA(ts, order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=forecast_days)
    forecast_dates = pd.date_range(data['ds'].iloc[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='B')
    return pd.DataFrame({'ds': forecast_dates, 'y': forecast})

def sarima_model(data, forecast_days):
    ts = data.set_index('ds')['y']
    model = SARIMAX(ts, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12))
    model_fit = model.fit(disp=False)
    forecast = model_fit.forecast(steps=forecast_days)
    forecast_dates = pd.date_range(data['ds'].iloc[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='B')
    return pd.DataFrame({'ds': forecast_dates, 'y': forecast})

def prophet_model(data, forecast_days):
    m = Prophet()
    m.fit(data)
    future = m.make_future_dataframe(periods=forecast_days)
    forecast = m.predict(future)
    forecast = forecast[['ds', 'yhat']].rename(columns={'yhat': 'y'})
    return forecast.tail(forecast_days)

def lstm_model(data, forecast_days):
    df_lstm = data.copy()
    df_lstm.set_index('ds', inplace=True)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_lstm[['y']])

    sequence_length = 60
    if len(scaled_data) <= sequence_length:
        raise ValueError(f"Not enough data for LSTM. Need at least {sequence_length + 1} records.")

    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i - sequence_length:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(64, activation='relu', return_sequences=True, input_shape=(sequence_length, 1)),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    input_seq = scaled_data[-sequence_length:].reshape((1, sequence_length, 1))
    forecast_scaled = []

    for _ in range(forecast_days):
        pred = model.predict(input_seq, verbose=0)[0, 0]
        forecast_scaled.append(pred)
        new_seq = np.append(input_seq[0, 1:, 0], pred)
        input_seq = new_seq.reshape((1, sequence_length, 1))

    forecast = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1)).flatten()

    last_date = pd.to_datetime(data['ds'].iloc[-1])
    forecast_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=forecast_days, freq='B')

    return pd.DataFrame({'ds': forecast_dates, 'y': forecast})

# ------------------ EVALUATION FUNCTION ------------------
def evaluate_model(model_func, data, forecast_days):
    if len(data) < forecast_days + 60:
        return None, None, None, None, None, None, None, None

    train = data[:-forecast_days]
    test = data[-forecast_days:]

    try:
        forecast = model_func(train.copy(), forecast_days)
        test['ds'] = pd.to_datetime(test['ds'])
        forecast['ds'] = pd.to_datetime(forecast['ds'])
        merged = pd.merge(test, forecast, on='ds', how='inner', suffixes=('_actual', '_pred'))

        if merged.empty:
            return None, None, None, None, None, None, None, None

        actual = merged['y_actual']
        pred = merged['y_pred']
        epsilon = 1e-8

        mae = mean_absolute_error(actual, pred)
        rmse = mean_squared_error(actual, pred, squared=False)
        mape = np.mean(np.abs((actual - pred) / (actual + epsilon))) * 100
        smape = np.mean(2 * np.abs(pred - actual) / (np.abs(actual) + np.abs(pred) + epsilon)) * 100
        r2 = r2_score(actual, pred)
        mbe = np.mean(pred - actual)
        medae = median_absolute_error(actual, pred)

        naive_forecast = actual.shift(1).dropna()
        mase = mae / np.mean(np.abs(actual.iloc[1:] - naive_forecast))

        return mae, rmse, mape, smape, r2, mbe, mase, medae
    except Exception as e:
        st.warning(f"Could not evaluate model: {e}")
        return None, None, None, None, None, None, None, None

# ------------------ HELPER: ADD DROPDOWN MENU ------------------
def add_timeframe_dropdown(fig):
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label="7D", step="day", stepmode="backward"),
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(step="all", label="All")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )
    return fig

# ------------------ HISTORICAL CHART ------------------
st.subheader(f"📊 Historical Stock Price for {ticker}")
fig_hist = go.Figure()
fig_hist.add_trace(go.Candlestick(
    x=df_full['Date'],
    open=df_full['Open'],
    high=df_full['High'],
    low=df_full['Low'],
    close=df_full['Close'],
    name='Candlestick'
))
fig_hist = add_timeframe_dropdown(fig_hist)
fig_hist.update_layout(
    title=f"Historical Price Data for {ticker}",
    xaxis_title="Date",
    yaxis_title="Price",
    hovermode="x unified"
)
st.plotly_chart(fig_hist, use_container_width=True)

# ------------------ FORECASTS ------------------
st.subheader(f"🔮 Forecasts for {unit_count} {time_unit}")

with st.spinner("Running all models..."):
    models = {
        "ARIMA": arima_model,
        "SARIMA": sarima_model,
        "Prophet": prophet_model,
        "LSTM": lstm_model
    }

    scores = []
    for name, func in models.items():
        try:
            forecast_df = func(df_model.copy(), forecast_days)

            fig = go.Figure()

            # Historical candles
            fig.add_trace(go.Candlestick(
                x=df_model['ds'],
                open=df_model['y'],
                high=df_model['y'],
                low=df_model['y'],
                close=df_model['y'],
                name='Historical'
            ))

            # Forecast candles
            fig.add_trace(go.Candlestick(
                x=forecast_df['ds'],
                open=forecast_df['y'],
                high=forecast_df['y'],
                low=forecast_df['y'],
                close=forecast_df['y'],
                name='Forecast',
                increasing_line_color='red',
                decreasing_line_color='red'
            ))

            # SMA & EMA lines
            fig.add_trace(go.Scatter(x=df_model['ds'], y=df_model['SMA_100'], mode='lines', name='100-day SMA', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=df_model['ds'], y=df_model['SMA_200'], mode='lines', name='200-day SMA', line=dict(color='green')))
            fig.add_trace(go.Scatter(x=df_model['ds'], y=df_model['EMA'], mode='lines', name='EMA', line=dict(color='purple')))

            fig = add_timeframe_dropdown(fig)
            fig.update_layout(title=f"{name} Forecast with Moving Averages", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

            # Evaluate
            mae, rmse, mape, smape, r2, mbe, mase, medae = evaluate_model(func, df_model.copy(), min(forecast_days, 60))
            if all(v is not None for v in [mae, rmse, mape, smape, r2, mbe, mase, medae]):
                scores.append({
                    'Model': name, 'MAE': mae, 'RMSE': rmse,
                    'MAPE (%)': mape, 'sMAPE (%)': smape,
                    'R²': r2, 'MBE': mbe, 'MASE': mase, 'MedAE': medae
                })
        except Exception as e:
            st.error(f"❌ Error running {name}: {e}")

# ------------------ PERFORMANCE TABLE ------------------
if scores:
    scores_df = pd.DataFrame(scores)
    scores_df['Forecast Accuracy (%)'] = 100 - scores_df['MAPE (%)']

    for metric in ['MAE', 'RMSE', 'MAPE (%)', 'sMAPE (%)', 'MASE', 'MedAE']:
        scores_df[f'{metric}_rank'] = scores_df[metric].rank()
    scores_df['Total_Rank'] = scores_df[[col for col in scores_df.columns if '_rank' in col]].sum(axis=1)

    scores_df = scores_df.sort_values(by='Total_Rank')
    best_model_name = scores_df.iloc[0]['Model']

    st.subheader("🏆 Model Performance (Last 60 Business Days)")
    display_cols = ['MAE', 'RMSE', 'MAPE (%)', 'sMAPE (%)', 'R²', 'MBE', 'MASE', 'MedAE', 'Forecast Accuracy (%)']
    st.dataframe(
        scores_df.set_index('Model')[display_cols].style.format({
            'MAE': '{:.2f}', 'RMSE': '{:.2f}',
            'MAPE (%)': '{:.2f}', 'sMAPE (%)': '{:.2f}',
            'R²': '{:.3f}', 'MBE': '{:.2f}', 'MASE': '{:.3f}', 'MedAE': '{:.2f}',
            'Forecast Accuracy (%)': '{:.2f}%'
        })
        .highlight_min(axis=0, subset=['MAE', 'RMSE', 'MAPE (%)', 'sMAPE (%)', 'MASE', 'MedAE'], color='#D4EDDA')
        .highlight_max(axis=0, subset=['R²', 'Forecast Accuracy (%)'], color='#D4EDDA')
    )
    st.success(f"✅ Based on a balanced ranking of all metrics, *{best_model_name}* is the most consistent model.")
else:
    st.warning("⚠ Could not evaluate model performance. This may be due to insufficient historical data.")