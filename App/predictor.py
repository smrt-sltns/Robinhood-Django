import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
from datetime import date


st.title('Stock Trend Prediction')


stock = st.text_input( label = 'Enter Stock Ticker', value = 'AAPL')
start_date = st.text_input( label = 'Start Date', value = '2010-01-01' )
end_date = st.text_input( label = 'End Date', value = date.today() )

df = data.DataReader( stock, 'yahoo', start_date, end_date )
df = df.drop(['Adj Close'], axis = 1)
st.subheader(f'Data from {start_date} to {end_date} ')

st.write(df.describe())

st.subheader('Closing Time vs Time Chart')
fig = plt.figure( figsize = (12,6) )
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Time vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure( figsize = (12,6) )
plt.plot(df.Close, 'b')
plt.plot(ma100, 'g')
st.pyplot(fig)

st.subheader('Closing Time vs Time Chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure( figsize = (12,6) )
plt.plot(df.Close, 'b' )
plt.plot(ma100, 'g' )
plt.plot(ma200, 'r' )
st.pyplot(fig)

data_training = pd.DataFrame( df[ 'Close' ][ 0 : int( len( df ) * 0.7 ) ] )
data_testing = pd.DataFrame( df[ 'Close' ][ int( len( df ) * 0.7 ) : int( len( df ) ) ] )

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler( feature_range = (0,1) )

data_training_array = scaler.fit_transform( data_training )

x_train = []
y_train = []

for i in range(100, data_training_array.shape[0]):
    x_train.append( data_training_array[i-100 : i] )
    y_train.append( data_training_array[i , 0] )

x_train, y_train = np.array(x_train), np.array(y_train)

model = load_model('keras_model.h5')

past_100_days = data_training.tail(100)
final_df = past_100_days.append( data_testing, ignore_index = True )
input_data = scaler.fit_transform(final_df)


x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append( input_data[i-100 : i] )
    y_test.append( input_data[i , 0] )

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

st.subheader( 'Predicted vs Original' )
fig2 = plt.figure( figsize = (12,6) )
plt.plot( y_test, 'b', label = 'Original Price' )
plt.plot( y_predicted, 'g', label = 'Predicted Price' )
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
