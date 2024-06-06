import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split


def read_data(data):
  """ Reads the data as a Pandas DataFrame. """
  df = pd.read_csv(data)
  try:
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.set_index('Datetime', inplace=True)
  except:
    pass
  return df


def lineplot(data):
  """ Visualizes the DataFrame. """
  fig = px.line(
    data, x=data.index, y=data["Close"],
    labels={'x': 'Date', 'y': 'Value'},
    title='Time Series Data'
  )
  st.plotly_chart(fig, use_container_width=True)


def peek_data(df):
  """ Shows 50 lines of the dataset on Streamlit Page. """
  with st.expander("Tabular"):
    showData = st.multiselect(
       'Filter: ', df.columns, default=[]
    )
    st.write(df[showData].head(50))


def prepare_dataset_lstm(dataset, time_steps=1):
  """ For timestamp currency. """
  X, y = [], []
  for i in range(len(dataset) - time_steps):
    X.append(dataset[i:(i + time_steps), 0])
    y.append(dataset[i + time_steps, 0])
  return np.array(X), np.array(y)


def lstm_train(df, target_col, train_size):
  """ Training an LSTM model and performing a forecast. """
  st.write("LSTM Training has begun.")
  scaler = MinMaxScaler(feature_range=(0, 1))
  scaled_data = scaler.fit_transform(df[target_col].values.reshape(-1, 1))
  train_size = int(len(scaled_data) * train_size)
  train, test = scaled_data[:train_size], scaled_data[train_size:]
  time_steps = 1
  X_train, y_train = prepare_dataset_lstm(train, time_steps)
  X_test, y_test = prepare_dataset_lstm(test, time_steps)
  # Reshape input to be [samples, time steps, features]
  X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
  X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
  model = Sequential()
  model.add(
    LSTM(
      units=50, return_sequences=True,
      input_shape=(X_train.shape[1], X_train.shape[2])
    )
  )
  model.add(LSTM(units=50))
  model.add(Dense(units=1))
  model.compile(optimizer='adam', loss='mean_squared_error')
  model.fit(X_train, y_train, epochs=20, batch_size=64, verbose=1)
  train_predict = model.predict(X_train)
  test_predict = model.predict(X_test)
  train_predict = scaler.inverse_transform(train_predict)
  y_train = scaler.inverse_transform([y_train])
  test_predict = scaler.inverse_transform(test_predict)
  y_test = scaler.inverse_transform([y_test])
  train_rmse = np.sqrt(mean_squared_error(y_train[0], train_predict[:,0]))
  test_rmse = np.sqrt(mean_squared_error(y_test[0], test_predict[:,0]))
  st.write(f"Train RMSE: {train_rmse}")
  st.write(f"Test RMSE: {test_rmse}")
  train_predict_plot = np.empty_like(scaled_data)
  train_predict_plot[:, :] = np.nan
  train_predict_plot[time_steps:len(train_predict) + time_steps, :] = train_predict
  test_predict_plot = np.empty_like(scaled_data)
  test_predict_plot[:, :] = np.nan
  start_idx = len(train_predict) + (time_steps * 2)
  end_idx = start_idx + len(test_predict)
  test_predict_plot[start_idx:end_idx, :] = test_predict
  fig = go.Figure()
  fig.add_trace(
    go.Scatter(
      x=df.index, y=df[target_col].values, mode='lines', name='Actual'
    )
  )
  fig.add_trace(
    go.Scatter(
      x=df.index[:len(train_predict_plot)], y=train_predict_plot.flatten(),
      mode='lines', name='Train Predictions', line=dict(color='blue')
      )
  )
  fig.add_trace(
    go.Scatter(
      x=df.index[:len(test_predict_plot)], y=test_predict_plot.flatten(),
      mode='lines', name='Test Predictions', line=dict(color='red')
    )
  )
    
  # Grafik ayarları
  fig.update_layout(
    title='Actual vs Predicted',
    xaxis_title='Date',
    yaxis_title=target_col,
    height=600,
    width=1000
  )
  st.plotly_chart(fig)
  return model


def st_app():
    """ Builds a streamlit app with user interface. """
    st.subheader("Welcome to the Stock Forecasting")
    st.sidebar.image("srock.jpeg", caption="stockforecast")
    stock_list = [
      "Amazon", "eBay", "Google", "IBM", "Meta", "Netflix",
      "PayPal"
    ]
    #st.write("Please choose a file and press the Upload button.")
    #uploaded_file = st.file_uploader("Dosya Seç", type=['csv'])
    target_stock = st.selectbox(
        "Please choose the company stock:", stock_list
    )
    if target_stock is not None:
      filename = str(target_stock).lower() + ".csv"
      df = read_data(filename)
      drop_nan = st.checkbox('Drop NaN values')
      if drop_nan: df = df.dropna()
      st.subheader("Line Plot")
      lineplot(df)
      st.subheader("Data Content")
      peek_data(df)
      train_size = st.slider(
      "Training size proportion:", min_value=0.0,
      max_value=1.0, step=0.01
      )
      target_col = st.selectbox(
        "Select the target column for time series data:", df.columns
      )
      if st.button("Train the model"):
        st.write("Model training is in progress...")
        lstm_train(df, target_col=target_col, train_size=train_size)
 

def main():
  st.set_page_config(page_title="Dashboard", page_icon="🐶", layout="wide")
  st_app()


if __name__ == '__main__':
    main()
