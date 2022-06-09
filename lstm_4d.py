'''
These functions are used to predicting moving weekly price ranges using daily information. 
The the target column is the last column in the dataframe, the rest columns are the features. 
'''

# Initial imports
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
# import hvplot.pandas
# imports for creating dataframe
from utils_laramie import get_df, get_all_raw_data

# imports for getting weekly range
from utils_laramie import calc_weekly_range

#imports for grouping data into weekly windows
from utils_laramie import grp_y_wk_d, drop_off_weeks

#shape data
from utils_laramie import get_X_y

def window_data(df, window, chunk_rows):
    X_list = df.iloc[:,:-1].values.tolist()
    X_chunks=[X_list[i:i + chunk_rows] for i in range(0, len(X_list), chunk_rows)]
    X = [X_chunks[i:i+window] for i in range (len(X_chunks)-window)]
    y_list=df.iloc[:,-1].values.tolist()
    y_chunks = [y_list[i + chunk_rows-1] for i in range(0, len(y_list), chunk_rows)]
    y = [y_chunks[i+window] for i in range(len(y_chunks)-window)]
    return np.array(X), np.array(y)

def data_splited_scaled(X,y):
    '''
    This function splits X and y into training,validation and testing sets, scales the data with MinMaxScaler and reshapes features data for the LSTM model .
    '''
    # Use 70% of the data for training and the remainder for testing
    split = int(0.7 * len(X))
    X_train = X[: split]
    split_1 = int(0.7*len(X_train))
    X_val = X_train[split_1:]
    X_train = X_train[:split_1]
    X_test = X[split:]
    y_train = y[: split]
    y_val = y_train[split_1:]
    y_train = y_train[:split_1]
    y_test = y[split:]

    # Create a MinMaxScaler object
    scaler = MinMaxScaler()

    # Fit the MinMaxScaler object with the training feature data X_train
    X_scaler = scaler.fit(X_train.ravel().reshape(-1,1))

    # Scale the features training and testing sets
    X_train_scaled= X_scaler.transform(X_train.ravel().reshape(-1,1))
    X_val_scaled= X_scaler.transform(X_val.ravel().reshape(-1,1))
    X_test_scaled = X_scaler.transform(X_test.ravel().reshape(-1,1))

    # Fit the MinMaxScaler object with the training target data y_train
    y_scaler = scaler.fit(y_train.ravel().reshape(-1,1))

    # Scale the target training and testing sets
    y_train_scaled = y_scaler.transform(y_train.ravel().reshape(-1,1))
    y_val_scaled = y_scaler.transform(y_val.ravel().reshape(-1,1))
    y_test_scaled = y_scaler.transform(y_test.ravel().reshape(-1,1))

    # Reshape the features for the model
    X_train_scaled = X_train_scaled.reshape((X_train.shape[:]))
    X_val_scaled = X_val_scaled.reshape((X_val.shape[:]))
    X_test_scaled = X_test_scaled.reshape((X_test.shape[:]))

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled, scaler

def lstm_model(X_train_scaled,number_units, optimizer, loss, metrics):
    '''
    This function builds a LSTM model

    '''
    # Define the LSTM RNN model.
    lstm_model = Sequential()

    dropout_fraction = 0.2
    # Layer 1
    lstm_model.add(LSTM(
    units=number_units,
    # return_sequences=True,
    input_shape=(X_train_scaled.shape[1],X_train_scaled.shape[2]*X_train_scaled.shape[3])
    ))
    lstm_model.add(Dropout(dropout_fraction))
    # # Layer 2
    # lstm_model.add(LSTM(units=number_units))
    # lstm_model.add(Dropout(dropout_fraction))
    # Output layer
    lstm_model.add(Dense(1))

    # Compile the lstm_model
    lstm_model.compile(optimizer= optimizer, loss=loss, metrics= metrics)

    return lstm_model


def model_fit(X_train_scaled, y_train_scaled,X_val_scaled, y_val_scaled, model,epochs):
    '''
    This function trains the lstm model
    '''
    history = model.fit(X_train_scaled.reshape(X_train_scaled.shape[0],X_train_scaled.shape[1],-1), y_train_scaled, validation_data=(X_val_scaled.reshape(X_val_scaled.shape[0],X_val_scaled.shape[1],-1), y_val_scaled), epochs=epochs, shuffle=False, batch_size=1, verbose=1)
    return (history, model)

def lstm_evaluation(X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, model_fitted):
    '''
    This function evaluates the LSTM model
    '''
    train_loss,train_mse,train_mae  = model_fitted.evaluate(X_train_scaled.reshape(X_train_scaled.shape[0],X_train_scaled.shape[1],-1), y_train_scaled,verbose=0)
    val_loss,val_mse,val_mae  = model_fitted.evaluate(X_val_scaled.reshape(X_val_scaled.shape[0],X_val_scaled.shape[1],-1), y_val_scaled,verbose=0)
    return train_loss,train_mse,train_mae,val_loss,val_mse,val_mae


def lstm_prediction(X_test_scaled, y_test_scaled,model_fitted,scaler):
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
    This function plots the actual weekly range vs. the predicted values.
    '''
    return df[["actual","predicted"]].plot(
                    ylabel= ylabel,
                    title=title
)
