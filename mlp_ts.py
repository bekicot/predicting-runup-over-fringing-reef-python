#%%
# Importing Library
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.backend import set_value
from sklearn.model_selection import cross_val_score, KFold
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasRegressor
from IPython.display import SVG, HTML, display
from tabulate import tabulate
from pathlib import Path
import keras.optimizers as optim
import numpy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import keras

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#%%
# Defining plot_hist and plot_comparison
def plot_hist(history):
    # Plot training & validation accuracy values
    plt.plot(history.history['loss'])
    plt.title('Model MSE')
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')


def plot_comparison(y_target, y_pred, output):
    # get_ipython().run_line_magic('matplotlib', 'inline')
    plt.style.use("seaborn-white")
    plt.rcParams['figure.figsize'] = [30, 10]
    plt.title(output)
    plt.xlabel("Data Points")
    plt.ylabel("Height")
    plt.plot(y_pred, 'r-', markersize=3, linewidth=1, label="Prediksi")
    plt.plot(y_target, 'b-', label="Observasi")
    plt.savefig(output)
    plt.legend()
    plt.close()

def coeff_determination(y_true, y_pred):
    from keras import backend as K
    SS_res = K.sum(K.square(y_true-y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))


#%%
# Preprocess Input Files
warnings.filterwarnings('ignore')

TEST_LOCATION = 'data/UMtests_1-53/test-18.dat'
pd_data = pd.read_csv(TEST_LOCATION, header=7,
                      usecols=numpy.arange(0, 10), sep="\t")
data = pd_data.to_numpy()
#%%
numpy.random.seed(2019)  # for reproducibility

# Best Epoch Based On:
#   ReduceLROnPlateau(monitor='loss', patience=10)
#   EarlyStopping(monitor='loss', verbose=1, patience=50)
header = [
    "Input Layer",
    "Output Layer",
    "Hidden Layer",
    "Neurons",
]

# pomm => Plain Old MLP Model (Input = X, Output = Y)
# mtsm => MLP Time Series Model (Input = [X-Lookback-1, X-1], Output = Y)
# model => "pomm", "mtsm"
# Data (X, Y)
def build_data(data, look_back=3, model="pomm"):
    if(model == "mtsm"):
        X = []
        Y = []
        for i in range(0, len(data[0])):
            if (i >= look_back):
                X.append(data[0][(i - look_back):i].flatten())
                Y.append(data[1][i])
    else:
        X = data[0]
        Y = data[1]

    assert(len(X) == len(Y))
    assert(len(X) > 0)
    # Split into 70%, 15%, 15%
    X_3_splits = numpy.split(X, [int(0.7 * len(X)), int(0.85 * len(X))])
    Y_3_splits = numpy.split(Y, [int(0.7 * len(X)), int(0.85 * len(X))])
    return (
        (X_3_splits[0], Y_3_splits[0]),
        (X_3_splits[1], Y_3_splits[1]),
        (X_3_splits[2], Y_3_splits[2]),
    )

x = data[400:, 0:1]
y = data[400:, 8:9]

scaler_x = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

x = scaler_x.fit_transform(x)
y = scaler_y.fit_transform(y)

(train_x, train_y), (val_x, val_y), (test_x, test_y) = build_data(
    (x, y), model="mtsm", look_back=1000)

model = Sequential()
model.add(Dense(300, input_dim=1000, activation='relu'))
model.add(Dense(300))
model.add(keras.layers.Dropout(0.1))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse',
             metrics=["mape", coeff_determination],
             optimizer=optim.SGD(lr=0.01))
history = model.fit(
    x=train_x,
    y=train_y,
    epochs=100,
    validation_split=0.1,
    callbacks=[
        # EarlyStopping(monitor='loss', verbose=1, patience=50),
        # keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=10)
    ],
    verbose=1
)

testing_evaluation = model.evaluate(x=test_x, y=test_y)

plot_comparison(
    scaler_y.inverse_transform(test_y),
    scaler_y.inverse_transform(model.predict(test_x)),
    '' + '6' + '.png'
)

plot_comparison(
    scaler_y.inverse_transform(train_y),
    scaler_y.inverse_transform(model.predict(train_x)),
    '' + '6_2' + '.png'
)

# model.save(MODEL_LOCATION)
