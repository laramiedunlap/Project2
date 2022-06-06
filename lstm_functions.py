# Initial imports
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import hvplot.pandas

# Set the random seed for reproducibility
# Note: This is used for model prototyping, but it is good practice to comment this out and run multiple experiments to evaluate your model.
from numpy.random import seed
seed(1)
from tensorflow import random

random.set_seed(2)


def window_data(df, window, feature_col_1, feature_col_2, target_col):
    '''
    This function accepts the column number for the features (X) and the target (y).
    It chunks the data up with a rolling window of Xt - window to predict Xt.
    It returns two numpy arrays of X and y.
    '''
    X = []
    y = []
    for i in range(len(df) - window):
        features = df.iloc[i : (i + window), feature_col_1:feature_col_2]
        target = df.iloc[(i + window), target_col]
        X.append(features)
        y.append(target)
    return np.array(X), np.array(y).reshape(-1, 1)


def data_splited_scaled(df, window, feature_col_1,feature_col_2, target_col):
    '''
    This function splits X and y into training and testing sets, scales the data with MinMaxScaler and reshapes features data for the LSTM model .
    '''  
    X, y = window_data(df, window, feature_col_1,feature_col_2, target_col)
    # Use 70% of the data for training and the remainder for testing
    split = int(0.7 * len(X))
    X_train = X[: split]
    X_test = X[split:]
    y_train = y[: split]
    y_test = y[split:]

    # Create a MinMaxScaler object
    scaler = MinMaxScaler()

    # Fit the MinMaxScaler object with the training feature data X_train
    scaler.fit(X_train.ravel().reshape(-1,1))

    # Scale the features training and testing sets
    X_train_scaled= scaler.transform(X_train.ravel().reshape(-1,1))
    X_test_scaled = scaler.transform(X_test.ravel().reshape(-1,1))

    # Fit the MinMaxScaler object with the training target data y_train
    scaler.fit(y_train)

    # Scale the target training and testing sets
    y_train_scaled = scaler.transform(y_train)
    y_test_scaled = scaler.transform(y_test)

    # Reshape the features for the model
    feature_num = feature_col_2 - feature_col_1
    X_train_scaled = X_train_scaled.reshape((X_train.shape[0], X_train.shape[1], feature_num))
    X_test_scaled = X_test_scaled.reshape((X_test.shape[0], X_test.shape[1], feature_num))

    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler


def lstm_model(df, window, feature_col_1, feature_col_2, target_col, number_units):
    '''
    This function builds and trains a 3-layer LSTM model
    '''
    X_train_scaled, _, y_train_scaled, _ ,_= data_splited_scaled(df, window,feature_col_1, feature_col_2, target_col)

    # Define the LSTM RNN model.
    lstm_model = Sequential()

    dropout_fraction = 0.2
    # calculate
    X_train_scaled, _, _, _ ,_= data_splited_scaled(df, window,feature_col_1, feature_col_2, target_col)
    # Layer 1
    feature_num = feature_col_2 - feature_col_1
    lstm_model.add(LSTM(
    units=number_units,
    return_sequences=True,
    input_shape=(X_train_scaled.shape[1], feature_num))
    )
    lstm_model.add(Dropout(dropout_fraction))
    # Layer 2
    lstm_model.add(LSTM(units=number_units, return_sequences=True))
    lstm_model.add(Dropout(dropout_fraction))
    # Layer 3
    lstm_model.add(LSTM(units=number_units))
    lstm_model.add(Dropout(dropout_fraction))
    # Output layer
    lstm_model.add(Dense(1))

    # Compile the lstm_model
    lstm_model.compile(optimizer="adam", loss="mean_squared_error")

    # Train the lstm_model
    lstm_model.fit(X_train_scaled, y_train_scaled, epochs=10, shuffle=False, batch_size=1, verbose=1)

    return lstm_model


def lstm_evaluation(df, window,feature_col_1, feature_col_2, target_col, number_units):
    '''
    This function evaluates the LSTM model
    '''
    _, X_test_scaled, _, y_test_scaled,_ =data_splited_scaled(df, window, feature_col_1, feature_col_2, target_col)
    model = lstm_model(df, window, feature_col_1, feature_col_2, target_col, number_units)
    score = model.evaluate(X_test_scaled, y_test_scaled,verbose=0)
    return score


def lstm_prediction(df, window, feature_col_1, feature_col_2, target_col, number_units):
    '''
    This function predicts y values and recover the original prices, and then creates a dataframe of Acural and Predicted values of y
    '''
    _, X_test_scaled, _, y_test_scaled,scaler =data_splited_scaled(df, window, feature_col_1, feature_col_2, target_col)
    model= lstm_model(df, window, feature_col_1, feature_col_2, target_col, number_units)
    y_predicted = model.predict(X_test_scaled)

    # Recover the original prices instead of the scaled version
    predicted_prices = scaler.inverse_transform(y_predicted)
    actual_prices = scaler.inverse_transform(y_test_scaled.reshape(-1, 1))

    prediction_df = pd.DataFrame({
        "Actual":actual_prices.ravel(),
        "Predicted":predicted_prices.ravel(),
    })

    return prediction_df