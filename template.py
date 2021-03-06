from keras.layers import LSTM
from keras.layers import Dense
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import numpy
import matplotlib.pyplot as mpt
import pandas

X_train = []
y_train = []

# Customizable Parameters 
# Good starting values are timestep = 60, epochs = 80, batch_size = 50
timestep = 1
epochs = 1
batch_size = 1

# Getting Data
dataset_train = pandas.read_csv('trainData.csv')
trainset = dataset_train.iloc[:, 1:2].values

# Scaling data down 
sc = MinMaxScaler(feature_range=(0, 1))
setScaled = sc.fit_transform(trainset)

for i in range(timestep, len(dataset_train)):
    X_train.append(setScaled[i-timestep:i, 0])
    y_train.append(setScaled[i, 0])
X_train, y_train = numpy.array(X_train), numpy.array(y_train)

X_train = numpy.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Building Models
def modelBuilder(model):
    model.add(LSTM(units=32, return_sequences=True,
                   input_shape=(X_train.shape[1], 1)))

    model.add(LSTM(units=48, return_sequences=True))
    model.add(LSTM(units=56, return_sequences=True))
    model.add(LSTM(units=60, return_sequences=True))
    model.add(LSTM(units=64))
    model.add(Dense(units=1))

    model.compile(optimizer='rmsprop', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    
def anotherBuilder(model):
    model.add(LSTM(units=32, return_sequences=True,
                   input_shape=(X_train.shape[1], 1)))

    model.add(LSTM(units=50, return_sequences=True))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)


mod = Sequential()
modelBuilder(mod)

dataset_test = pandas.read_csv('testData.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price 
dataset_total = pandas.concat(
    (dataset_train['Open'], dataset_test['Open']), axis=0)
val = dataset_total[len(dataset_total) -
                    len(dataset_test) - timestep:].values
val = val.reshape(-1, 1)
val = sc.transform(val)
X_test = []

for i in range(timestep, (timestep+len(dataset_test))):
    X_test.append(val[i-timestep:i, 0])
X_test = numpy.array(X_test)
X_test = numpy.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = mod.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Graphing the results
mpt.plot(real_stock_price, color=(0, 0.9, 0.1), label='Stock Price')
mpt.plot(predicted_stock_price, color=(0, 0.6, 0.8),
         label='Predicted Stock Price')
mpt.title('Predicted Price')
mpt.xlabel('Time')
mpt.ylabel('Real Price')
mpt.legend()
