#YPAI06-Capstone2
Time Series. Covid 19 cases prediction

CAPSTONE 2
Covid 19 cases
Import packages import os import datetime import IPython import IPython.display import matplotlib as mpl import matplotlib.pyplot as plt import numpy as np import pandas as pd import seaborn as sns import tensorflow as tf from tensorflow import keras from keras.models import Sequential from keras.layers import LSTM, Dense, Dropout from time_series_helper import WindowGenerator
mpl.rcParams['figure.figsize'] = (8, 6) mpl.rcParams['axes.grid'] = False 2. Data loading df = pd.read_csv('cases_malaysia_covid.csv') selected_columns = ['date', 'cases_new', 'cases_import', 'cases_recovered', 'cases_active'] df = df[selected_columns]

date_time = pd.to_datetime(df.pop('date'), format='%d/%m/%Y') dataset_train = pd.read_csv('cases_malaysia_covid.csv') dataset_train.head() df.info() training_set = dataset_train.iloc[:,1:2].values

print(training_set) print(training_set.shape) df['cases_new'] = df['cases_new'].replace({',': ''}, regex=True)

df['cases_new'] = pd.to_numeric(df['cases_new'], errors='coerce', downcast='integer')

print(df.dtypes) df.describe().T dataset_test = pd.read_csv('cases_malaysia_covid.csv') actual_stock_price = dataset_test.iloc[:,1:2].values

Data inspection
df.set_index(date_time, inplace=True) plot_cols = ['cases_new', 'cases_import', 'cases_recovered', 'cases_active'] plot_features = df[plot_cols] plot_features.index = date_time _ = plot_features.plot(subplots=True)

plot_features = df[plot_cols][:480] plot_features.index = date_time[:480] _ = plot_features.plot(subplots=True)

Data cleaning
print(df.isnull().sum()) #cases_new have null value Data cases images

Box plot
df.boxplot(figsize=(12, 8)) plt.show() df.hist(figsize=(20,20), edgecolor='black') plt.show() df['cases_new'].fillna(df['cases_new'].median(), inplace=True) print(df.isnull().sum()) print(df.duplicated().sum()) print(df.shape) df.drop_duplicates(inplace=True)

Double check duplicate
print(df.duplicated().sum()) print(df.shape)

Train, validation, test split for time series data
column_indices = {name: i for i, name in enumerate(df.columns)}

n = len(df) train_df = df[0:int(n0.7)] val_df = df[int(n0.7):int(n0.9)] test_df = df[int(n0.9):]

num_features = df.shape[1]

Data normalization
train_mean = train_df.mean() train_std = train_df.std()

train_df = (train_df - train_mean) / train_std val_df = (val_df - train_mean) / train_std test_df = (test_df - train_mean) / train_std df_std = (df - train_mean) / train_std df_std = df_std.melt(var_name='Column', value_name='Normalized') plt.figure(figsize=(12, 6)) ax = sns.violinplot(x='Column', y='Normalized', data=df_std) _ = ax.set_xticklabels(df.keys(), rotation=90)

one_predict = WindowGenerator(input_width=25, label_width=25, shift=1, train_df=train_df, val_df=val_df, test_df=test_df, label_columns=['cases_new'])# %% lstm_model = keras.Sequential() lstm_model.add(keras.layers.LSTM(128, return_sequences=True)) lstm_model.add(keras.layers.Dropout(0.2)) lstm_model.add(keras.layers.Dense(1))

Function to create sequences for the LSTM models
def create_sequences(data, window_size, output_width, offset): X, y = [], [] for i in range(len(data) - window_size - output_width + 1): X.append(data[i:i + window_size]) y.append(data[i + window_size + offset - 1:i + window_size + output_width + offset - 1, 0]) # Predicting 'cases_new'

# Convert lists of sequences to NumPy arrays
X = np.array(X)

# Concatenate y sequences into a 2D NumPy array
y = np.array(y)
y = np.vstack(y)  # Stack sequences vertically

return X, y
MAX_EPOCHS = 40

def compile_and_fit(model, window, patience=3): early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min')

model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(), metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanAbsolutePercentageError()])

history = model.fit(window.train, epochs=MAX_EPOCHS, validation_data=window.val, callbacks=[early_stopping])

Compile the model and train
history_1 = compile_and_fit(lstm_model, one_predict)

Evaluate the model
print(lstm_model.evaluate(one_predict.val)) print(lstm_model.evaluate(one_predict.test))

Plot the resultt
one_predict.plot(model=lstm_model, plot_col='cases_new') Multistep window multi_predict = WindowGenerator(input_width=25, label_width=25, shift=25, train_df=train_df, val_df=val_df, test_df=test_df, label_columns=['cases_new']) window_size = 30 output_width = 30 offset_single_step = 1 offset_multi_step = 30 #Build step model multi_lstm = keras.Sequential() multi_lstm.add(keras.layers.LSTM(32, return_sequences=False)) multi_lstm.add(keras.layers.Dense(25*1)) multi_lstm.add(keras.layers.Reshape([25,1]))

Convert the DataFrame to a numpy array
data = df[['cases_new', 'cases_import', 'cases_recovered', 'cases_active']].values

Create sequences for single step window scenario
X_single, y_single = create_sequences(data, window_size, output_width, offset_single_step)

Define the LSTM model for single step window scenario
model_single = Sequential() model_single.add(LSTM(64, input_shape=(window_size, data.shape[1]))) model_single.add(Dropout(0.2)) model_single.add(Dense(output_width)) model_single.compile(loss='mae', optimizer='adam')

Compile and train model for multi step
history_2 = compile_and_fit(multi_lstm, multi_predict)

Evaluate the model
print(multi_lstm.evaluate(multi_predict.val)) print(multi_lstm.evaluate(multi_predict.test))

Plot the resultt
multi_predict.plot(model=lstm_model, plot_col='cases_new') from datetime import datetime

log_dir_one_predict = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "_one_predict" log_dir_multi_predict = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "_multi_predict"

Create WindowGenerators
one_predict = WindowGenerator(input_width=25, label_width=25, shift=1, train_df=train_df, val_df=val_df, test_df=test_df, label_columns=['cases_new']) multi_predict = WindowGenerator(input_width=25, label_width=25, shift=25, train_df=train_df, val_df=val_df, test_df=test_df, label_columns=['cases_new'])

Callback for TensorBoard
tensorboard_callback_one_predict = tf.keras.callbacks.TensorBoard(log_dir=log_dir_one_predict, histogram_freq=1) tensorboard_callback_multi_predict = tf.keras.callbacks.TensorBoard(log_dir=log_dir_multi_predict, histogram_freq=1)

Compile and fit for one-step prediction
compile_and_fit(lstm_model, one_predict)

Pass TensorBoard callback in the fit function
history_1 = lstm_model.fit(one_predict.train, epochs=MAX_EPOCHS, validation_data=one_predict.val, callbacks=[tensorboard_callback_one_predict])

Compile and fit for multi-step prediction
compile_and_fit(multi_lstm, multi_predict)

Pass TensorBoard callback in the fit function
history_2 = multi_lstm.fit(multi_predict.train, epochs=MAX_EPOCHS, validation_data=multi_predict.val, callbacks=[tensorboard_callback_multi_predict])
