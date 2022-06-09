# Initial imports
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Dropout, Flatten
# import hvplot.pandas
# imports for creating dataframe
from utils_laramie import get_df, get_all_raw_data

# imports for getting weekly range
from utils_laramie import calc_weekly_range

#imports for grouping data into weekly windows
from utils_laramie import grp_y_wk_d, drop_off_weeks

#shape data
from utils_laramie import get_X_y

def window_data(df, window, chunk_rows, feature_col_1, feature_col_2, target_col):
    X_list = df.iloc[:,feature_col_1:feature_col_2].values.tolist()
    X_chunks=[X_list[i:i + chunk_rows] for i in range(0, len(X_list), chunk_rows)]
    X = [X_chunks[i:i+window] for i in range (len(X_chunks)-window)]
    y_list=df.iloc[:,target_col].values.tolist()
    y_chunks = [y_list[i + chunk_rows-1] for i in range(0, len(y_list), chunk_rows)]
    y = [y_chunks[i+window] for i in range(len(y_chunks)-window)]
    return np.array(X), np.array(y)

def data_splited_scaled(X,y,):
    '''
    This function splits X and y into training and testing sets, scales the data with MinMaxScaler and reshapes features data for the LSTM model .
    '''
    # Use 70% of the data for training and the remainder for testing
    split = int(0.7 * len(X))
    X_train = X[: split]
    X_test = X[split:]
    y_train = y[: split]
    y_test = y[split:]

    # Create a MinMaxScaler object
    scaler = MinMaxScaler()

    # Fit the MinMaxScaler object with the training feature data X_train
    X_scaler = scaler.fit(X_train.ravel().reshape(-1,1))

    # Scale the features training and testing sets
    X_train_scaled= X_scaler.transform(X_train.ravel().reshape(-1,1))
    X_test_scaled = X_scaler.transform(X_test.ravel().reshape(-1,1))

    # Fit the MinMaxScaler object with the training target data y_train
    y_scaler = scaler.fit(y_train.ravel().reshape(-1,1))

    # Scale the target training and testing sets
    y_train_scaled = y_scaler.transform(y_train.ravel().reshape(-1,1))
    y_test_scaled = y_scaler.transform(y_test.ravel().reshape(-1,1))

    # Reshape the features for the model
    feature_num = feature_col_2 - feature_col_1
    X_train_scaled = X_train_scaled.reshape((X_train.shape[0], X_train.shape[1],X_train.shape[2], feature_num))
    X_test_scaled = X_test_scaled.reshape((X_test.shape[0], X_test.shape[1],X_train.shape[2], feature_num))

    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler

def lstm_model(number_units):
    '''
    This function builds and trains a 3-layer LSTM model

    '''
    # Define the LSTM RNN model.
    lstm_model = Sequential()

    dropout_fraction = 0.2
    # Layer 1
    lstm_model.add(LSTM(
    units=number_units,
    return_sequences=True,
    input_shape=(window, chunk_rows*(feature_col_2-feature_col_1))
    ))
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

    return lstm_model


def model_fit(X_train_scaled, y_train_scaled,model):
    history = model.fit(X_train_scaled.reshape(X_train_scaled.shape[0],X_train_scaled.shape[1],-1), y_train_scaled, epochs=100, shuffle=False, batch_size=1, verbose=1)
    return (history, model)

def lstm_evaluation(model_fitted):
    '''
    This function evaluates the LSTM model
    '''
    score = model_fitted.evaluate(X_test_scaled.reshape(X_test_scaled.shape[0],X_test_scaled.shape[1],-1), y_test_scaled,verbose=0)
    return score


def lstm_prediction(model_fitted):
    '''
    This function predicts y values and recover the original prices, and then creates a dataframe of Acural and Predicted values of y
    '''
    y_predicted = model_fitted.predict(X_test_scaled.reshape(X_test_scaled.shape[0],X_test_scaled.shape[1],-1))

    # Recover the original prices instead of the scaled version
    predicted = scaler.inverse_transform(y_predicted)
    actual = scaler.inverse_transform(y_test_scaled.reshape(-1, 1))

    prediction_df = pd.DataFrame({
        "actual":actual.ravel(),
        "predicted":predicted.ravel(),
    })

    return prediction_df


def predicted_plotting(df, ylabel,title):
    '''
    This function plots the actual prices vs. the predicted values.
    '''
    return df[["actual","predicted"]].plot(
                    ylabel= ylabel,
                    title=title
)