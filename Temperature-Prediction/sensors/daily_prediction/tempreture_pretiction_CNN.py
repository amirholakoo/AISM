import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ReduceLROnPlateau
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
data = pd.read_csv('Today.csv')
values = data['Temperature'].values  
values = np.array(values).reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(-1, 1))
temperature = scaler.fit_transform(values)
joblib.dump(scaler, 'scaler.save')
timestamps = pd.to_datetime(data['Time steps'])
time_minutes = timestamps.dt.hour * 60 + timestamps.dt.minute
time_sin = np.sin(2 * np.pi * time_minutes / 1440) 
time_cos = np.cos(2 * np.pi * time_minutes / 1440)
features = np.stack([temperature.reshape(-1), time_sin, time_cos], axis=1)
# Create sequences
def create_data_set_multi(features, past_steps=1440, future_steps=1440, steps = 60 ):
    X, Y = [], []
     # because 60 minutes = 1 hour

    for i in range(len(features) - past_steps - future_steps + 1):
        # Past 1440 minutes (1 day) as input
        X.append(features[i:i+past_steps])  
        # Next 1440 minutes, but sampled every 60 minutes → 24 values
        Y.append(features[i+past_steps : i+past_steps+future_steps : steps, 0])
    
    X = np.array(X)
    Y = np.array(Y)  
    return X, Y

X, Y = create_data_set_multi(features, past_steps=1440, future_steps=1440, steps=60)


# # Random train/test split
# train_size = int(0.8 * len(X))
# x_train, x_test = X[:train_size], X[train_size:]
# y_train, y_test = Y[:train_size], Y[train_size:]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, shuffle=True, random_state=20)


n_features = x_train.shape[2]   
future_steps = y_train.shape[1] 

# -----------------------------
# Define 1D-CNN model
# -----------------------------
model = Sequential([
    Conv1D(128, kernel_size=2, activation='relu', strides=1, padding='same',input_shape=(x_train.shape[1], n_features)),
    Conv1D(128, kernel_size=2, strides=1, padding='same', activation='relu'),
    Conv1D(128, kernel_size=2 , strides=1, padding='same', activation = 'relu'),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dense(future_steps, activation='linear')])
learning_rate = 0.001
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='mse')


# -----------------------------
# Train the model
# -----------------------------
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.99,      
    patience=10,      
    min_lr=1e-6,      
    verbose=1
)

history = model.fit(
    x_train, y_train,
    epochs=1000,
    batch_size=64,
    validation_data=(x_test, y_test),
    callbacks = [reduce_lr]
)
model.save("tempreture_pretiction_CNN.h5")
y_train_pred_scaled= model.predict(x_train)  
y_test_pred_scaled= model.predict(x_test)
def inverse_transform_y(y_scaled, scaler):
    
    y_2d = y_scaled.reshape(-1, 1)               # flatten to 2D
    y_inv = scaler.inverse_transform(y_2d)       # inverse scaling
    return y_inv.reshape(y_scaled.shape)         # reshape back

y_train_true = inverse_transform_y(y_train, scaler)
y_test_true  = inverse_transform_y(y_test, scaler)
y_train_pred = inverse_transform_y(y_train_pred_scaled, scaler)
y_test_pred  = inverse_transform_y(y_test_pred_scaled, scaler)

# -----------------------------
# Evaluation metrics
# -----------------------------
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))

def r2(y_true, y_pred):
    return r2_score(y_true.flatten(), y_pred.flatten())

print("Train RMSE:", rmse(y_train_true, y_train_pred))
print("Test RMSE:", rmse(y_test_true, y_test_pred))
print("Train R²:", r2(y_train_true, y_train_pred))
print("Test R²:", r2(y_test_true, y_test_pred))

# -----------------------------
# Plot predictions vs true values for first sample
# -----------------------------
plt.figure(figsize=(12,5))
plt.plot(y_test_true[0], label='True Temperature', color='r')
plt.plot(y_test_pred[0], label='Predicted Temperature', color='b')
plt.xlabel("Future Time Steps")
plt.ylabel("Temperature (°C)")
plt.title("Temperature Prediction - First Test Sample")
plt.legend()
plt.show()

# Optional: plot multiple samples
plt.figure(figsize=(12,5))
for i in range(5):
    plt.plot(y_test_true[i], color='r', alpha=0.5)
    plt.plot(y_test_pred[i], color='b', alpha=0.5)
plt.xlabel("Future Time Steps")
plt.ylabel("Temperature (°C)")
plt.title("Temperature Prediction - First 5 Test Samples")
plt.show()