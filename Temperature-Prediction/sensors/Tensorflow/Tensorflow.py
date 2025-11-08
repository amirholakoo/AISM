import datetime 
import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import pickle

mpl.rcParams['figure.figsize'] = (8,6)
mpl.rcParams['axes.grid'] = False

# Load and preprocess data
df = pd.read_csv('DHT22_data.csv')
date_time = pd.to_datetime(df.pop('Time steps'), unit='s').dt.floor('s')
timestamps_s = date_time.map(pd.Timestamp.timestamp)
print(timestamps_s[0])
# Time periods
second = 1
minute = 60 * second
hour = 60 * minute
day = 24 * hour
month = 30 * day

# Add cyclical time features
df['Month sin'] = np.sin(2*np.pi * timestamps_s / month)
df['Month cos'] = np.cos(2*np.pi * timestamps_s / month)
df['Day sin'] = np.sin(2*np.pi * timestamps_s / day)
df['Day cos'] = np.cos(2*np.pi * timestamps_s / day)
df['Hour sin'] = np.sin(2*np.pi * timestamps_s / hour)
df['Hour cos'] = np.cos(2 * np.pi * timestamps_s / hour)
df['Minute sin'] = np.sin(2*np.pi * timestamps_s / minute)
df['Minute cos'] = np.cos(2*np.pi * timestamps_s / minute)

# Split data
column_indices = {name: i for i, name in enumerate(df.columns)}
n = len(df)
train_df = df[0:int(n*0.70)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]
num_features = df.shape[1]

# Normalize data
train_mean = train_df.mean()
train_std = train_df.std()
train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

# Save scaler stats
scaler_stats = {'mean': train_mean, 'std': train_std}
with open('scaler_stats.pkl', 'wb') as f:
    pickle.dump(scaler_stats, f)

# WindowGenerator class
class WindowGenerator():
    def __init__(self, input_width, label_width, shift, 
                 train_df=train_df, val_df=val_df, test_df=test_df, 
                 label_columns=None):
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
    
    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'
        ])

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]

        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns], 
                axis=-1
            )
        if self.label_columns is not None:
            labels.set_shape([None, self.label_width, len(self.label_columns)])  # ✅ [batch, 24, 1]
        else:
            labels.set_shape([None, self.label_width, num_features])
        return inputs, labels

    def make_dataset(self, data, shuffle=True):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=shuffle,
            batch_size=64
        )
        ds = ds.map(self.split_window)
        if shuffle:
            ds = ds.cache().shuffle(buffer_size=1000)
        ds = ds.prefetch(10)
        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df, shuffle=True)

    @property
    def val(self):
        return self.make_dataset(self.val_df, shuffle=False)

    @property
    def test(self):
        return self.make_dataset(self.test_df, shuffle=False)

    @property
    def example(self):
        result = getattr(self, '_example', None)
        if result is None:
            result = next(iter(self.train))
            self._example = result
        return result

    def plot(self, model=None, plot_col='Temperature', max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        
        for n in range(max_n):
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                    label='Inputs', marker='.', zorder=-10)
            
            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                       edgecolors='k', label='Labels', c='#2ca02c', s=64)
            
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                           marker='X', edgecolors='k', label='Predictions', 
                           c='#ff7f0e', s=64)
                if n == 0:
                    plt.legend()

        plt.xlabel('Time [h]')
        plt.show()

# Create single-step window for example
w = WindowGenerator(input_width=6, label_width=1, shift=1, label_columns=['Temperature'])
print(w)

# Create example window
example_window = tf.stack([
    np.array(train_df[:w.total_window_size]),
    np.array(train_df[100:100+w.total_window_size]),
    np.array(train_df[200:200+w.total_window_size])
])

example_inputs, example_labels = w.split_window(example_window)
w._example = example_inputs, example_labels  # Store as instance variable

print('All shapes are: (batch, time, features)')
print(f'Window shape: {example_window.shape}')
print(f'Inputs shape: {example_inputs.shape}')
print(f'Labels shape: {example_labels.shape}')

# Training functionu
Max_Epochs = 1000
def compile_and_fit(model, window, patience=20):
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=optimizer,
                  metrics=[tf.keras.metrics.MeanAbsoluteError()])
    
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.99,
        patience=patience,
        verbose=1,
        min_lr=1e-10
    )
    
    history = model.fit(window.train, epochs=Max_Epochs, 
                       validation_data=window.val) 
                       #callbacks=[lr_scheduler])
    return history

# Multi-step prediction setup
multi_val_performance = {}
multi_performance = {}

OUT_STEPS = 24
multi_window = WindowGenerator(input_width=24, label_width=OUT_STEPS, shift=OUT_STEPS,label_columns=['Temperature'] )
multi_window.plot()

# Feedback LSTM Model

class FeedBack(tf.keras.Model):
    def __init__(self, units, out_steps):
        super().__init__()
        self.out_steps = out_steps
        self.units = units
        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
        self.dense = tf.keras.layers.Dense(num_features)

    def warmup(self, inputs):
        x, *state = self.lstm_rnn(inputs)
        prediction = self.dense(x)
        return prediction, state

    def call(self, inputs, training=None):
        predictions = []
        prediction, state = self.warmup(inputs)
        predictions.append(prediction)

        for n in range(1, self.out_steps):
            x = prediction
            x, state = self.lstm_cell(x, states=state, training=training)
            prediction = self.dense(x)
            predictions.append(prediction)

        predictions = tf.stack(predictions)
        predictions = tf.transpose(predictions, [1, 0, 2])
        return predictions

# Create and train model
feedback_model = FeedBack(units=1000, out_steps=OUT_STEPS)

# Test warmup
print("Testing warmup...")
prediction, state = feedback_model.warmup(multi_window.example[0])
print(f"Warmup prediction shape: {prediction.shape}")

# Train model
print("Training model...")
history = compile_and_fit(feedback_model, multi_window)

# Clear output and evaluate
IPython.display.clear_output()

multi_val_performance['AR LSTM'] = feedback_model.evaluate(multi_window.val, return_dict=True)
multi_performance['AR LSTM'] = feedback_model.evaluate(multi_window.test, verbose=0, return_dict=True)

# Plot results
multi_window.plot(feedback_model)

# Print performance
print("Validation Performance:")
for name, perf in multi_val_performance.items():
    print(f"{name:<12} loss: {perf['loss']:.4f}, mae: {perf['mean_absolute_error']:.4f}")

print("\nTest Performance:")
for name, perf in multi_performance.items():
    print(f"{name:<12} loss: {perf['loss']:.4f}, mae: {perf['mean_absolute_error']:.4f}")

    # Additional block to add after training and evaluation
# This will plot predictions vs actuals for validation and test data
# Using 3 random batches from each

import random

# Function to plot predictions on given dataset
def plot_predictions(window, model, dataset, title, max_subplots=3):
    # Get a few batches from the dataset
    all_inputs = []
    all_labels = []
    all_predictions = []
    
    # Collect 3 batches
    dataset_iter = iter(dataset)
    for _ in range(max_subplots):
        try:
            inputs, labels = next(dataset_iter)
            predictions = model(inputs)
            all_inputs.append(inputs)
            all_labels.append(labels)
            all_predictions.append(predictions)
        except StopIteration:
            break
    
    # Concatenate for plotting
    inputs = tf.concat(all_inputs, axis=0)
    labels = tf.concat(all_labels, axis=0)
    predictions = tf.concat(all_predictions, axis=0)
    
    plt.figure(figsize=(12, 8))
    plot_col = 'Temperature'  # Assuming we plot Temperature; change if needed
    plot_col_index = window.column_indices[plot_col]
    
    if window.label_columns:
        label_col_index = window.label_columns_indices.get(plot_col, None)
    else:
        label_col_index = plot_col_index
    
    if label_col_index is None:
        print(f"Column {plot_col} not found for labeling.")
        return
    
    max_n = min(max_subplots, len(inputs))
    for n in range(max_n):
        plt.subplot(max_n, 1, n+1)
        plt.ylabel(f'{plot_col} [normed]')
        plt.plot(window.input_indices, inputs[n, :, plot_col_index],
                 label='Inputs', marker='.', zorder=-10)
        
        # Plot actual labels
        plt.scatter(window.label_indices, labels[n, :, label_col_index],
                    edgecolors='k', label='Actual', c='#2ca02c', s=64)
        
        # Plot predictions
        plt.scatter(window.label_indices, predictions[n, :, label_col_index],
                    marker='X', edgecolors='k', label='Predictions', c='#ff7f0e', s=64)
        
        if n == 0:
            plt.legend()
    
    plt.xlabel('Time [h]')
    plt.suptitle(title)
    plt.show()

# After training and evaluation, add this:
print("Plotting Validation Data Predictions")
plot_predictions(multi_window, feedback_model, multi_window.val, title='Validation Data: Predictions vs Actual')

print("Plotting Test Data Predictions")
plot_predictions(multi_window, feedback_model, multi_window.test, title='Test Data: Predictions vs Actual')


feedback_model.save_weights("feedback_lstm.weights.h5")
print("✅ Model weights saved successfully.")