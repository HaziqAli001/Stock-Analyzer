import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import streamlit as st 
import datetime as dt


start = '2010-01-01'
#end = '2022-12-31'
end = dt.datetime.now()

st.title('Stock Data App')

user_input = st.text_input('Enter the stock Ticker','AAPL')
data = yf.Ticker(user_input)
new = data.history(start=start, end = end)

#Details of Company
st.subheader('Company Details')
st.write('Name: ' + data.info['shortName'])
st.write('Latest Price: $' + str(round(new.Close.tail(1)[0],3)))

st.subheader('Major Holders')
st.write(data.major_holders)

st.subheader('Institutional Holders')
st.write(data.institutional_holders)

st.subheader('Recommendations')
st.write(data.recommendations)

#Description
st.subheader('Data from 2010 to now')
st.write(new.describe())

#Visualization
st.subheader('Closing Price vs Time')
fig = plt.figure(figsize=(12,6))
plt.plot(new.Close)
st.pyplot(fig)


st.subheader('Closing Price vs Time with 100 Moving Average')
ma100 = new.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
st.pyplot(fig)

st.subheader('Closing Price vs Time with 100 and 200 Moving Average')
ma100 = new.Close.rolling(100).mean()
ma200 = new.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(ma200)
st.pyplot(fig)

#Model Data
training_data = pd.DataFrame(new['Close'][0:int(len(new)*0.7)])
testing_data = pd.DataFrame(new['Close'][int(len(new)*0.7):int(len(new))])
print(training_data.shape)
print(testing_data.shape)

scalar = MinMaxScaler(feature_range=(0,1))

data_training_array = scalar.fit_transform(training_data)


#Feeding Data into model

model =load_model('price_predictor.h5')

past100days = training_data.tail(100)
final_df = pd.concat([past100days,testing_data],ignore_index=True)
input_data = scalar.fit_transform(final_df)
x_test = []
y_test = []
for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
    
x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scalar.scale_
y_test = y_test/scaler

y_predicted = y_predicted/scaler

#Plotting the predictions
st.subheader('The Predicted Prices')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label ='Original Price')
plt.plot(y_predicted,'r',label ='Predicted Price')
st.pyplot(fig2)

