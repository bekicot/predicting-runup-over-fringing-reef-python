
#%%
# %matplotlib osx
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Lambda
from IPython.display import SVG, HTML, display
from tabulate import tabulate
from pathlib import Path
import keras.optimizers as optim
import numpy

import pandas as pd

df = pd.DataFrame

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [25, 15]

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
MODEL_LOCATION = 'model_1_test.h5'
TEST_LOCATION = 'data/UMtests_1-53/test-17.dat'
# import matplotlib.pyplot as plt

#%%
MODEL_LOCATION = 'RNN.h5'
numpy.random.seed(7)
#%%
def genfromfile(path):
    from numpy import genfromtxt

    file = open(path, mode="r+")
    for _ in range(8):
        next(file)
    return genfromtxt(file, dtype=float)

def transform_to_timesteps(data, look_back, features_min, features_max):
    x = []
    y = []
    # timestep terakhir mulai dari panjang data di kurang panjang history
    for i in range(len(data) - 1 - look_back):
        x.append(data[i:(i + look_back), features_min:features_max])
        # 9 Adalah index runup (mulai dari 0)
        y.append(data[i + look_back][9])
    return numpy.array(x, numpy.float), numpy.array(y, numpy.float)

def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

#%%
scaler_y = MinMaxScaler(feature_range=(0, 1))
scaler_x = MinMaxScaler(feature_range=(0, 1))
gauge_min = 0
gauge_max = 1
look_back = 300
data_source = genfromfile("data/UMtests_1-53/test-16.dat")
data_source[:, gauge_min:gauge_max] = scaler_x.fit_transform(data_source[:, gauge_min:gauge_max])
data_source[:, 8:9] = scaler_y.fit_transform(data_source[:, 8:9])
x, y = transform_to_timesteps(data_source, look_back, gauge_min, gauge_max)
x_train = x[0:17000]
y_train = y[0:17000]


#%%
# print(genfromfile("data/UMtests_1-53/test-1.dat")[0])
if Path(MODEL_LOCATION).is_file():
    model = Sequential()
    model.add(LSTM(32, input_shape=(300,1), kernel_initializer="lecun_uniform"))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    history = model.fit(x=x_train, y=y_train, epochs=1000)
    model.save(MODEL_LOCATION)
    with open("model_1.json", 'w') as save_location:
        save_location.write(model.to_json())
else:
    model = load_model(MODEL_LOCATION)


#%%
plt.close('all')
test_min = 3000
test_max = 5000
result = model.predict(x=x[test_min:test_max])

original_y = scaler_y.inverse_transform(numpy.reshape(y, (len(y), 1)))[test_min:test_max]
inverse_result = scaler_y.inverse_transform(result)
data = genfromfile("data/UMtests_1-53/test-16.dat")

plt.figure(2)
plt.xlabel("Data Points")
plt.ylabel("Height")
plt.plot(inverse_result, 'r-', markersize=3, linewidth=1)
plt.plot(data[3000:5000, 8:9], 'b-', markersize=3, linewidth=1)

#%%



