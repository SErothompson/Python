import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Generate a sine wave dataset
def create_sine_wave_dataset(timesteps):
    x = np.linspace(0, 50, timesteps)
    y = np.sin(x)
    return y

timesteps = 100
data = create_sine_wave_dataset(timesteps)

# Prepare the data for LSTM (many-to-one)
def prepare_data(data, n_steps):
    X, y = [], []
    for i in range(len(data)):
        end_ix = i + n_steps
        if end_ix > len(data)-1:
            break
        seq_x, seq_y = data[i:end_ix], data[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

n_steps = 10
X, y = prepare_data(data, n_steps)

# Reshape data for LSTM [samples, timesteps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train the LSTM model
model.fit(X, y, epochs=200, verbose=0)

# Make a prediction
x_input = data[-n_steps:]
x_input = x_input.reshape((1, n_steps, 1))
yhat = model.predict(x_input, verbose=0)
print(f"Predicted next value: {yhat[0][0]}")