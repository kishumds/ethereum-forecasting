import matplotlib.pyplot as plt
import streamlit as st
from datetime import date
import pandas as pd
from PIL import Image
import pickle


st.sidebar.title("About")
st.sidebar.info("Forecasting Close Price of Ethereum using 'NeuralProphet' Machine Learning model.")

def get_input():
	st.sidebar.header("Input From user")
	st.sidebar.subheader("Select range of Date for visualize data for particular date range.")
	st.sidebar.write("(From 13-09-2015 to 13-09-2021)")
	start_date = st.sidebar.text_input("Start Date", "13-09-2015")
	end_date = st.sidebar.text_input("End Date", "13-09-2021")
	st.write("")
	st.sidebar.subheader("Enter Period for Forecasting of Price")
	period = st.sidebar.text_input("Period (In Days)", "30")
	return start_date, end_date, period

image = Image.open('eth.png')
st.image(image, use_column_width=True)

START = "13-09-2015"
TODAY = date.today().strftime("%d-%M-%Y")

st.title('Ethereum Price Forecasting')

def get_data(start, end):
	df = pd.read_csv('ETH-USD.csv')

	start = pd.to_datetime(start)
	end = pd.to_datetime(end)

	start_row = 0
	end_row = 0

	for i in range(0, len(df)):
		if start <=	pd.to_datetime(df['Date'][i]):
			start_row = i
			break

	for j in range(0, len(df)):
		if end >= pd.to_datetime(df['Date'][len(df)-1-j]):
			end_row = len(df) - 1 - j
			break

	df = df.set_index(pd.DatetimeIndex(df['Date'].values))

	return df.iloc[start_row:end_row+1, :]

start, end, period = get_input()
data = get_data(start, end)

st.subheader("Data")

st.write("First 5 Columns")
st.write(data.head())
st.write("Last 5 Columns")
st.write(data.tail())

st.subheader('Close Price')
st.write("Zoom In/Zoom Out for better visualization.")
st.line_chart(data[['Open', 'Close']])

st.subheader("Volume")
st.write("Zoom In/Zoom Out for better visualization.")
st.line_chart(data['Volume'])

st.subheader("Data Statistics")
st.write(data.describe())

st.header("Prediction")

def model_np():
	m = pickle.load(open('neuralProphet.pkl', 'rb'))

	st.subheader("Using NeuralProphet")
	df = data.copy()
	df.reset_index(inplace=True)
	df_train = df[['Date','Close']]
	df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})


	future = m.make_future_dataframe(df_train, periods=int(period))
	forecast = m.predict(future)
	forecast = forecast.rename(columns={"ds": "Date", "yhat1": "Close"})
	st.write("Forecasting of Etheruem Close Price from 14-09-2021 to 18-09-2021")
	st.write(forecast[['Date', 'Close']].head())
	st.write(f"Forecasting of Close Price of {period} days")
	st.line_chart(forecast['Close'])

model_np()