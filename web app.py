import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import seaborn as sns
from keras.models import load_model
import streamlit as st
#from yahoofinancials import YahooFinancials
from datetime import datetime
import yfinance as yf
import pickle


startdate = datetime(2010, 1, 29)
enddate= datetime.today()

st.title('Stock Trend Prediction')

user_input=st.text_input('Enter Stock Ticker', 'AAPL')
df=yf.download(user_input ,start=startdate, end=enddate)

#Describing Data
st.subheader('Data from 2010-2024')
st.write(df.describe())

# Visualization
st.subheader('Closing Price vs Time Chart')
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

# !00 days moving average
st.subheader('Closing Price vs Time chart with 100MA')
ma100=df.Close.rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100)
st.pyplot(fig)

#200 days moving average
st.subheader('Closing Price vs Time chart with 200MA')
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(ma200)
st.pyplot(fig)

# make separate dataframe of closed price
closedf = df['Close']
closedf.shape

# Splitting Data into training and testing
from sklearn.preprocessing import MinMaxScaler
#del closedf['Date'] 
scaler=MinMaxScaler()#(feature_range=(0,1))
close_stock=scaler.fit_transform(np.array(closedf).reshape(-1,1))
print(close_stock.shape)

# Ratio for training and testing data is 65:35
training_size=int(len(close_stock)*0.65)
test_size=len(close_stock)-training_size
train_data,test_data=close_stock[0:training_size,:],close_stock[training_size:len(close_stock),:1]
print("train_data: ", train_data.shape)
print("test_data: ", test_data.shape)

# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

# reshape into X=t,t+1,t+2,t+3,.....,t+14 and Y=t+15 
time_step = 15
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

print("X_train: ", X_train.shape)
print("y_train: ", y_train.shape)
print("X_test: ", X_test.shape)
print("y_test", y_test.shape)

# reshape input value X from 2D array to  3d array which is required 
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

print("X_train: ", X_train.shape)
print("X_test: ", X_test.shape)


#train_data=pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
#test_data=pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

# Scaling train data
#from sklearn.preprocessing import MinMaxScaler
#scaler=MinMaxScaler(feature_range=(0,1))
#data_training=scaler.fit_transform(train_data)


#Loading model
loaded_model = pickle.load(open("C:/Users/Mandar/Downloads/Project File/My project/Sarimax_model.sav",'rb'))
#model=load_model(r"C:/Users/Mandar/Downloads/Project File/My project/Sarimax_model.sav")
# testing part
#past_100_days=train_data.tail(100)
#final_df=np.concatenate((past_100_days, test_data), axis=0)
#input_data=scaler.fit_transform(final_df)

#x_test=[]
#y_test=[]

#for i in range(100, input_data.shape[0]):
#  x_test.append(input_data[i-100: i])
 # y_test.append(input_data[i, :])
#x_test, y_test = np.array(x_test),np.array(y_test)

#y_predicted=model.predict(x_test)
#y_predicted_final = y_predicted[:, -1, :]
# Prediction of next '10' days 
n_forecast = 10  
Prediction = loaded_model.get_forecast(steps=n_forecast)


# Extract forecasted values and confidence intervals
Prediction_values = Prediction.predicted_mean
#conf_int = forecast.conf_int()
# Create a time index for the forecasted values
#forecast_index = pd.date_range(start=pd.Timestamp.now().date(), periods=len(Prediction_values), freq= M)

# Create a DataFrame with the forecasted values 
Prediction_df = pd.DataFrame({'Prediction' : Prediction_values,})#, index=Prediction_values.index)
    
next_predicted_days_value= scaler.inverse_transform(np.array(Prediction_df).reshape(-1,1)).reshape(1,-1).tolist()[0]
st.subheader('Next 10 Days Values')
st.write(next_predicted_days_value)
# Visualization Actual and Predicted price.
#st.subheader('Predicted and Original')
sarimaxdf=close_stock.tolist()
sarimaxdf.extend((np.array(Prediction_df).reshape(-1,1)).tolist())
sarimaxdf=scaler.inverse_transform(sarimaxdf).reshape(1,-1).tolist()[0]

from itertools import cycle
import plotly.express as px
names = cycle(['Close price'])
fig = px.line(sarimaxdf,labels={'value': 'Stock price','index': 'Timestamp'})
fig.update_layout(title_text='Plotting Whole closing stock price with prediction',
                  plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='Stock')

fig.for_each_trace(lambda t:  t.update(name = next(names)))

fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()

# Visualization Actual vs Predicted.
#st.subheader('Predicted vs Original')
#fig2=plt.figure(figsize=(12,6))
#plt.plot(y_test,'b',label='Original Price')
#plt.plot(y_predicted_final,'r',label='Predicted Price')
#plt.title('Actual vs Predicted price')
#plt.xlabel('Time')
#plt.ylabel('Price')
#plt.legend()
#st.pyplot(fig2)
