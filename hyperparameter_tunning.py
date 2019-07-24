#%%
# Importing Library
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

#%%
# Defining build_model
def build_model(hidden_layers=1, neurons_each_layer=20, input_dim=None):
    model = Sequential()
    model.add(Dense(
        neurons_each_layer,
        input_dim=input_dim,
        kernel_initializer='random_uniform',
        activation='relu'))
    for _ in range(0, hidden_layers - 1):        
        model.add(Dense(
            neurons_each_layer,
            kernel_initializer='random_uniform',
            activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(
        loss='mse',
        metrics=["mape", coeff_determination],
        optimizer=optim.SGD(lr=0.001)
    )
    return model

def coeff_determination(y_true, y_pred):
    from keras import backend as K
    SS_res = K.sum(K.square(y_true-y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))
#%%
# Preprocess Input Files
import warnings
warnings.filterwarnings('ignore')

TEST_LOCATION = 'data/UMtests_1-53/test-18.dat'
pd_data = pd.read_csv(TEST_LOCATION, header=7, usecols=numpy.arange(0, 10), sep="\t")
data = pd_data.to_numpy()
#%%
# Run Analisa (BEHOLD!!!)
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
EPOCH = [50, 100, 500, 1000]
# EPOCH = [20, 80, 150, 200]
for fold in ["Train ", "Val "]:
    for epoch in EPOCH:
        for metric in ["MSE", "Mape", "R^2"]:
            header.append(fold + 'E' + str(epoch) + ' ' + metric)

for metric in ["MSE", "Mape", "R^2"]:
    header.append("Test " + metric)
results = [header]
nama_konfigurasi = "Hypertuning_Timeseries"

# Prepare Backup
pd.DataFrame(results).to_csv(
    '~' + nama_konfigurasi + ".csv")

# pomm => Plain Old MLP Model
# mtsm
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
_input_layers = [[0, 1]]
_output_layers = [[8, 9], [9, 10]]
_hidden_layers = range(1, 5)
_neurons = [100, 300]

# Hidden layer 1..3
for input_layers in _input_layers:
    input_name = nama_konfigurasi + '_' + \
        str(input_layers[0]) + ',' + str(input_layers[1]) + 'I_'
    # 9:10 => Runup Gauge
    # 8:9 => Perbandingan Zijlema
    for output_layer in _output_layers:
        # 0:1 => 15000++ non-function mapping
        output_name = input_name + str(output_layer[1]) + 'O_'
        x = data[100:, input_layers[0]:input_layers[1]]
        y = data[100:, output_layer[0]:output_layer[1]]

        scaler_x = MinMaxScaler(feature_range=(0, 1))
        scaler_y = MinMaxScaler(feature_range=(0, 1))

        x = scaler_x.fit_transform(x)
        y = scaler_y.fit_transform(y)

        for look_back in [20, 100, 200]:
            (train_x, train_y), (val_x, val_y), (test_x, test_y) = build_data((x, y), model="mtsm", look_back=look_back)

            for num_of_hidden_layer in _hidden_layers:  # Jangan lupa update nama konfigurasi
                hidden_name = output_name + str(num_of_hidden_layer) + 'H_'
                # Neuron 5, 10, 20, 30
                for num_of_neuron in _neurons:
                    neuron_name = hidden_name + str(num_of_neuron) + 'N'
                    # model = build_model(hidden_layers=num_of_hidden_layer, neurons_each_layer=num_of_neuron, input_dim=input_layers[1] - input_layers[0])
                    model = build_model(
                        hidden_layers=num_of_hidden_layer, neurons_each_layer=num_of_neuron, input_dim=look_back)
                    display("Memulai analisa dengen model: H" + neuron_name)
                    history = model.fit(
                        x=train_x,
                        y=train_y,
                        epochs=EPOCH[-1],
                        validation_split=0.1,
                        callbacks=[
                            # EarlyStopping(monitor='loss', verbose=1, patience=50),
                            # keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=10)
                        ],
                        verbose=1
                    )
                    testing_evaluation = model.evaluate(x=test_x, y=test_y)

                    result = [
                        ','.join(str(x) for x in input_layers),
                        output_layer[1],
                        num_of_hidden_layer,
                        num_of_neuron,
                    ]
                    for fold in ["", "val_"]:
                        for epoch in EPOCH:
                            for metric in ["loss", "mean_absolute_percentage_error", "coeff_determination"]:
                                result.append(
                                    history.history[fold + metric][epoch-1]
                                )
                    result.append(testing_evaluation[0])
                    result.append(testing_evaluation[1])
                    result.append(testing_evaluation[2])

                    plot_comparison(
                        scaler_y.inverse_transform(test_y),
                        scaler_y.inverse_transform(model.predict(test_x)),
                        'plots/' + neuron_name + '.png'
                    )

                    display(pd.DataFrame(results[1:], columns=results[0]))
                    keras.utils.plot_model(
                        model,
                        to_file="models/" + neuron_name + '.png',
                        show_shapes=True,
                        show_layer_names=False,
                        rankdir='LR'
                    )
                    # Backup Each Iteration
                    with open('~' + nama_konfigurasi + '.csv', "a") as csv_file:
                        csv_file.write(pd.DataFrame([result]).to_csv())
                    results.append(result)
                    # Simpan Weight Ke JSON
                    pd.Series(model.get_weights()).to_json('models/' + neuron_name + '.json', orient='values')
                    display("Selesai")

pd.DataFrame(results[1:], columns=results[0]).to_csv(
    nama_konfigurasi + ".csv")
pd.DataFrame(results[1:], columns=results[0])
# model.save(MODEL_LOCATION)
