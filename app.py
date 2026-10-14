import random
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

WINDOW = 60          # số ngày dùng để dự báo ngày kế tiếp
FORECAST_DAYS = 10   # số ngày dự báo
SEED = 42

st.set_page_config(page_title="Dự báo cổ phiếu LSTM", page_icon="📈", layout="wide")


# ---------------------------------------------------------------- DỮ LIỆU
def load_prices(ticker: str) -> pd.DataFrame:
    """Tải giá đóng cửa. Nếu chỉ gõ mã (VD: VNM) thì tự thêm đuôi .VN."""
    candidates = [ticker] if "." in ticker else [f"{ticker}.VN", ticker]
    end = date.today() + timedelta(days=1)
    for symbol in candidates:
        df = yf.download(symbol, start="2018-01-01", end=end.isoformat(),
                         progress=False, auto_adjust=False)
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):  # yfinance bản mới
                df.columns = df.columns.get_level_values(0)
            df = df[["Close"]].dropna()
            df.attrs["symbol"] = symbol
            return df
    return pd.DataFrame()


def create_sequences(data, window_size=WINDOW):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        y.append(data[i])
    return np.array(X), np.array(y)


# ---------------------------------------------------------------- MÔ HÌNH
@st.cache_data(ttl=6 * 3600, show_spinner=False)
def run_forecast(ticker: str):
    """Huấn luyện LSTM và dự báo 10 ngày. Kết quả được lưu cache 6 giờ."""
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import LSTM, Dense, Input
    from tensorflow.keras.models import Sequential

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    df = load_prices(ticker)
    if df.empty:
        return None
    symbol = df.attrs.get("symbol", ticker)
    if len(df) < 300:
        return {"error": f"Mã {symbol} chỉ có {len(df)} phiên giao dịch, "
                         "không đủ dữ liệu để huấn luyện (cần ít nhất 300)."}

    data = df[["Close"]].values
    dates = df.index

    train_size = int(0.8 * len(data))
    train_data, test_data = data[:train_size], data[train_size:]

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)

    X_train, y_train = create_sequences(train_scaled)
    test_combined = np.vstack((train_scaled[-WINDOW:], test_scaled))
    X_test, y_test = create_sequences(test_combined)

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    val_size = int(0.1 * len(X_train))
    X_val, y_val = X_train[-val_size:], y_train[-val_size:]
    X_fit, y_fit = X_train[:-val_size], y_train[:-val_size]

    model = Sequential([
        Input(shape=(WINDOW, 1)),
        LSTM(50, return_sequences=True),
        LSTM(50),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")

    early_stop = EarlyStopping(monitor="val_loss", patience=5,
                               restore_best_weights=True)
    model.fit(X_fit, y_fit, validation_data=(X_val, y_val),
              epochs=10, batch_size=32, shuffle=False,
              callbacks=[early_stop], verbose=0)

    # Đánh giá trên tập test
    test_pred = scaler.inverse_transform(model.predict(X_test, verbose=0))
    y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))
    metrics = {
        "RMSE": float(np.sqrt(mean_squared_error(y_test_real, test_pred))),
        "MAE": float(mean_absolute_error(y_test_real, test_pred)),
        "R2": float(r2_score(y_test_real, test_pred)),
    }

    # Dự báo 10 ngày tới (cuốn chiếu: dùng kết quả hôm trước làm đầu vào hôm sau)
    last_window = np.vstack((train_scaled, test_scaled))[-WINDOW:].reshape(1, WINDOW, 1)
    future = []
    for _ in range(FORECAST_DAYS):
        nxt = model.predict(last_window, verbose=0)
        future.append(nxt[0, 0])
        last_window = np.append(last_window[:, 1:, :], nxt.reshape(1, 1, 1), axis=1)
    future = scaler.inverse_transform(np.array(future).reshape(-1, 1)).flatten()

    # Ngày dự báo: chỉ tính ngày làm việc (thứ 2 - thứ 6), chưa trừ ngày lễ
    future_dates = pd.bdate_range(dates[-1] + pd.Timedelta(days=1), periods=FORECAST_DAYS)

    return {
        "symbol": symbol,
        "dates": dates,
        "close": data.flatten(),
        "test_dates": dates[train_size:],
        "test_pred": test_pred.flatten(),
        "future_dates": future_dates,
        "future": future,
        "metrics": metrics,
    }


# ---------------------------------------------------------------- GIAO DIỆN
st.title("📈 Dự báo giá cổ phiếu 10 ngày tới (LSTM)")
st.caption("Nhập mã cổ phiếu (VD: VNM, FPT, VCB, HPG...) rồi bấm **Dự báo**. "
           "Với mã nước ngoài hãy gõ đúng mã Yahoo Finance (VD: AAPL, TSLA).")

with st.form("input_form"):
    col1, col2 = st.columns([3, 1])
    ticker_input = col1.text_input("Mã cổ phiếu", value="VNM").strip().upper()
    submitted = col2.form_submit_button("Dự báo", use_container_width=True)

if submitted:
    if not ticker_input:
        st.warning("Vui lòng nhập mã cổ phiếu.")
        st.stop()

    with st.spinner("Đang tải dữ liệu và huấn luyện mô hình (khoảng 1-3 phút lần đầu)..."):
        try:
            result = run_forecast(ticker_input)
        except Exception as e:  # noqa: BLE001
            st.error(f"Có lỗi xảy ra: {e}")
            st.stop()

    if result is None:
        st.error(f"Không tìm thấy dữ liệu cho mã '{ticker_input}'. "
                 "Hãy kiểm tra lại mã cổ phiếu.")
        st.stop()
    if "error" in result:
        st.error(result["error"])
        st.stop()

    symbol = result["symbol"]
    last_price = float(result["close"][-1])
    last_date = result["dates"][-1]
    future = result["future"]
    change = (future[-1] / last_price - 1) * 100

    st.subheader(f"Kết quả cho {symbol}")
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Giá đóng cửa gần nhất ({last_date:%d/%m/%Y})", f"{last_price:,.0f}")
    c2.metric(f"Giá dự báo ngày {result['future_dates'][-1]:%d/%m/%Y}",
              f"{future[-1]:,.0f}", f"{change:+.2f}%")
    c3.metric("R² trên tập kiểm tra", f"{result['metrics']['R2']:.3f}")

    # Biểu đồ
    n_show = 250
    hist_dates = result["dates"][-n_show:]
    hist_close = result["close"][-n_show:]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_dates, y=hist_close, mode="lines",
                             name="Giá thật", line=dict(color="#1f77b4")))
    mask = result["test_dates"] >= hist_dates[0]
    fig.add_trace(go.Scatter(x=result["test_dates"][mask],
                             y=result["test_pred"][mask], mode="lines",
                             name="Dự báo trên tập test",
                             line=dict(color="#ff7f0e", dash="dot")))
    fx = [last_date] + list(result["future_dates"])
    fy = [last_price] + list(future)
    fig.add_trace(go.Scatter(x=fx, y=fy, mode="lines+markers",
                             name="Dự báo 10 ngày tới",
                             line=dict(color="#d62728"), marker=dict(size=7)))
    fig.update_layout(height=520, hovermode="x unified",
                      xaxis_title="Ngày", yaxis_title="Giá đóng cửa",
                      legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, use_container_width=True)

    # Bảng dự báo
    st.subheader("Bảng dự báo 10 ngày tới")
    table = pd.DataFrame({
        "Ngày": [d.strftime("%d/%m/%Y") for d in result["future_dates"]],
        "Giá dự báo": [f"{v:,.0f}" for v in future],
    })
    table.index = range(1, FORECAST_DAYS + 1)
    st.dataframe(table, use_container_width=True)

    with st.expander("Chỉ số đánh giá mô hình (tập kiểm tra)"):
        m = result["metrics"]
        st.write(f"- **RMSE:** {m['RMSE']:,.2f}")
        st.write(f"- **MAE:** {m['MAE']:,.2f}")
        st.write(f"- **R²:** {m['R2']:.4f}")

    st.info("⚠️ Đây là sản phẩm học thuật dùng mô hình LSTM chỉ dựa trên giá đóng cửa "
            "quá khứ, không xét tin tức, báo cáo tài chính hay yếu tố thị trường. "
            "Dự báo cuốn chiếu nhiều ngày có sai số tích lũy. "
            "**Không phải khuyến nghị đầu tư.**")
