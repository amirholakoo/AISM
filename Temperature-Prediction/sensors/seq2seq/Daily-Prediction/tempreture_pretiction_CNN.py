import tensorflow as tf
from tensorflow.keras.layers import Add, Input, Conv1D, Flatten, Dense, MultiHeadAttention
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib

data = pd.read_csv('Modified_Timeseries_Temperature_and_Humidity_1_Hour.csv')
temp = data['Temperature'].values.reshape(-1, 1)
timestamps = pd.to_datetime(data['Time steps'])
time_minutes = (timestamps.dt.hour * 60 + timestamps.dt.minute).to_numpy()
time_sin = np.sin(2 * np.pi * time_minutes / 1440).reshape(-1, 1)
time_cos = np.cos(2 * np.pi * time_minutes / 1440).reshape(-1, 1)

# Normalize temp directly (no sine)
temp_min = temp.min()
temp_max = temp.max()
temp_norm = (temp - temp_min) / (temp_max - temp_min)

# Optional: Smooth temp_norm if needed (but probably unnecessary)
# window_size = 5
# temp_norm = np.convolve(temp_norm.flatten(), np.ones(window_size)/window_size, mode='same').reshape(-1, 1)

features = np.concatenate([temp_norm, time_sin, time_cos], axis=1)
scaling_info = {'temp_min': temp_min, 'temp_max': temp_max}
joblib.dump(scaling_info, 'temp_scaler.pkl')

def create_data_set_multi(features, past_steps, future_steps):
    X, Y = [], []
    for i in range(len(features) - past_steps - future_steps + 1):
        X.append(features[i:i + past_steps])
        Y.append(temp_norm[i + past_steps : i + past_steps + future_steps].flatten())  # Flatten to (future_steps,)
    return np.array(X), np.array(Y)

past_steps = 24
future_steps = 24
X, Y = create_data_set_multi(features, past_steps, future_steps)

# Chronological split (no shuffle)
train_size = int(0.8 * len(X))
x_train, x_test = X[:train_size], X[train_size:]
y_train, y_test = Y[:train_size], Y[train_size:]

n_features = x_train.shape[2]

inputs = Input(shape=(past_steps, n_features))
x = Conv1D(128, kernel_size=2, activation='relu', strides=1, padding='same')(inputs)
x = Conv1D(64, kernel_size=2, strides=1, padding='same', activation='relu')(x)
x = Conv1D(16, kernel_size=2, strides=1, padding='same', activation='relu')(x)
attention_output = MultiHeadAttention(num_heads=4, key_dim=16)(query=x, value=x)
attention_output = Add()([x, attention_output])
x = Flatten()(attention_output)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)
outputs = Dense(future_steps, activation='linear')(x)  # Predict norm directly

model = Model(inputs=inputs, outputs=outputs)
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.99, patience=10, min_lr=1e-6, verbose=1)

history = model.fit(
    x_train, y_train,
    epochs=1000,
    batch_size=64,
    validation_data=(x_test, y_test),
    callbacks=[reduce_lr]
)
model.save("temperature_prediction_CNN.h5")

# Predictions and denormalize
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)
y_test_true = y_test * (temp_max - temp_min) + temp_min
y_test_pred_denorm = y_test_pred * (temp_max - temp_min) + temp_min

# Plot denormalized
plt.plot(y_test_true.flatten(), label='True')
plt.plot(y_test_pred_denorm.flatten(), label='Prediction')
plt.legend()
plt.show()