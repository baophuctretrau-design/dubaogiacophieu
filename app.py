import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import datetime

# 1. Cấu hình trang web
st.set_page_config(page_title="Dự Báo Cổ Phiếu LSTM", layout="wide")
st.title("📈 Ứng Dụng Dự Báo Giá Cổ Phiếu Bằng AI (LSTM)")
st.write("Nhập mã cổ phiếu để hệ thống tự động tải dữ liệu, huấn luyện mô hình và dự báo giá trong 10 ngày tới.")

# 2. Hộp nhập mã cổ phiếu
ticker = st.text_input("Nhập mã cổ phiếu (VD: VNM.VN, FPT.VN, VCB.VN, AAPL):", "VNM.VN")

if st.button("Dự đoán"):
    with st.spinner(f"Đang tải dữ liệu và huấn luyện mô hình cho {ticker}... (Quá trình này có thể mất 1-2 phút)"):
        try:
            # 3. Lấy dữ liệu chứng khoán
            end_date = datetime.date.today().strftime("%Y-%m-%d")
            df = yf.download(ticker, start="2018-01-01", end=end_date)
            
            if df.empty:
                st.error("Không tìm thấy dữ liệu! Hãy đảm bảo bạn nhập đúng mã cổ phiếu (Thêm hậu tố .VN đối với cổ phiếu Việt Nam).")
            else:
                data = df[['Close']].values
                
                # Chia tập Train/Test
                train_size = int(0.8 * len(data))
                train_data = data[:train_size]
                test_data = data[train_size:]
                
                scaler = MinMaxScaler()
                train_scaled = scaler.fit_transform(train_data)
                test_scaled = scaler.transform(test_data)
                
                def create_sequences(data, window_size=60):
                    X, y = [], []
                    for i in range(window_size, len(data)):
                        X.append(data[i-window_size:i])
                        y.append(data[i])
                    return np.array(X), np.array(y)
                
                X_train, y_train = create_sequences(train_scaled)
                test_combined = np.vstack((train_scaled[-60:], test_scaled))
                X_test, y_test = create_sequences(test_combined)
                
                X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
                X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
                
                val_size = int(0.1 * len(X_train))
                X_val = X_train[-val_size:]
                y_val = y_train[-val_size:]
                X_train_final = X_train[:-val_size]
                y_train_final = y_train[:-val_size]
                
                # 4. Xây dựng & Huấn luyện mô hình
                model = Sequential()
                model.add(LSTM(50, return_sequences=True, input_shape=(60, 1)))
                model.add(LSTM(50))
                model.add(Dense(1))
                model.compile(optimizer='adam', loss='mean_squared_error')
                
                early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
                
                # Rút gọn epoch xuống 5 để web không bị treo do xử lý quá lâu
                model.fit(X_train_final, y_train_final, validation_data=(X_val, y_val), 
                          epochs=5, batch_size=32, shuffle=False, callbacks=[early_stop], verbose=0)
                
                # 5. Dự báo 10 ngày tới
                last_60_days = np.vstack((train_scaled, test_scaled))[-60:]
                last_60_days = last_60_days.reshape(1, 60, 1)
                future_predictions = []
                
                for i in range(10):
                    next_pred = model.predict(last_60_days, verbose=0)
                    future_predictions.append(next_pred[0,0])
                    last_60_days = np.append(last_60_days[:,1:,:], next_pred.reshape(1,1,1), axis=1)
                    
                future_predictions = np.array(future_predictions).reshape(-1,1)
                future_predictions = scaler.inverse_transform(future_predictions)
                
                # 6. Hiển thị kết quả lên Web
                st.success(f"Đã huấn luyện và dự báo thành công cho {ticker}!")
                st.subheader(f"Dự báo giá 10 ngày tới")
                
                # Hiển thị bảng
                last_date = df.index[-1]
                future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, 11)]
                df_future = pd.DataFrame({
                    'Ngày': [d.strftime('%Y-%m-%d') for d in future_dates], 
                    'Giá dự báo': future_predictions.flatten()
                })
                st.table(df_future)
                
                # Vẽ biểu đồ
                st.subheader("Biểu đồ dự báo so với thực tế")
                y_test_real = scaler.inverse_transform(y_test.reshape(-1,1))
                test_pred = model.predict(X_test, verbose=0)
                test_pred = scaler.inverse_transform(test_pred)
                
                fig, ax = plt.subplots(figsize=(12, 5))
                # Chỉ hiển thị 150 ngày cuối của test set cho dễ nhìn
                ax.plot(y_test_real[-150:], label="Giá thật")
                ax.plot(test_pred[-150:], label="Mô hình Test dự báo")
                
                future_x = range(len(y_test_real[-150:]), len(y_test_real[-150:]) + 10)
                ax.plot(future_x, future_predictions, marker='o', color='red', label="Dự báo 10 ngày tới")
                
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
                
        except Exception as e:
            st.error(f"Đã xảy ra lỗi: {e}")
