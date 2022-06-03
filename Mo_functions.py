{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Initial imports\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "from pathlib import Path\n",
    "\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Set the random seed for reproducibility\n",
    "# Note: This is used for model prototyping, but it is good practice to comment this out and run multiple experiments to evaluate your model.\n",
    "from numpy.random import seed\n",
    "\n",
    "seed(1)\n",
    "from tensorflow import random\n",
    "\n",
    "random.set_seed(2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "def window_data(df, window, feature_col_number, target_col_number):\n",
    "    X = []\n",
    "    y = []\n",
    "    for i in range(len(df) - window):\n",
    "        features = df.iloc[i : (i + window), feature_col_number]\n",
    "        target = df.iloc[(i + window), target_col_number]\n",
    "        X.append(features)\n",
    "        y.append(target)\n",
    "    return np.array(X), np.array(y).reshape(-1, 1)\n",
    "    \n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "def lstm_fit(number_units):  \n",
    "    X, y = window_data(df, window_size, feature_column, target_column)  \n",
    "    # Use 70% of the data for training and the remainder for testing\n",
    "    split = int(0.7 * len(X))\n",
    "    X_train = X[: split]\n",
    "    X_test = X[split:]\n",
    "    y_train = y[: split]\n",
    "    y_test = y[split:]\n",
    "\n",
    "    # Use the MinMaxScaler to scale data between 0 and 1.\n",
    "    from sklearn.preprocessing import MinMaxScaler\n",
    "\n",
    "    # Create a MinMaxScaler object\n",
    "    scaler = MinMaxScaler()\n",
    "\n",
    "    # Fit the MinMaxScaler object with the training feature data X_train\n",
    "    scaler.fit(X_train)\n",
    "\n",
    "    # Scale the features training and testing sets\n",
    "    X_train = scaler.transform(X_train)\n",
    "    X_test = scaler.transform(X_test)\n",
    "\n",
    "    # Fit the MinMaxScaler object with the training target data y_train\n",
    "    scaler.fit(y_train)\n",
    "\n",
    "    # Scale the target training and testing sets\n",
    "    y_train = scaler.transform(y_train)\n",
    "    y_test = scaler.transform(y_test)\n",
    "\n",
    "    # Reshape the features for the model\n",
    "    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))\n",
    "    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))\n",
    "\n",
    "    # Import required Keras modules\n",
    "    from tensorflow.keras.models import Sequential\n",
    "    from tensorflow.keras.layers import LSTM, Dense, Dropout\n",
    "\n",
    "    # Define the LSTM RNN model.\n",
    "    model = Sequential()\n",
    "\n",
    "    dropout_fraction = 0.2\n",
    "\n",
    "# Layer 1\n",
    "    model.add(LSTM(\n",
    "    units=number_units,\n",
    "    return_sequences=True,\n",
    "    input_shape=(X_train.shape[1], 1))\n",
    "    )\n",
    "    model.add(Dropout(dropout_fraction))\n",
    "# Layer 2\n",
    "    model.add(LSTM(units=number_units, return_sequences=True))\n",
    "    model.add(Dropout(dropout_fraction))\n",
    "# Layer 3\n",
    "    model.add(LSTM(units=number_units))\n",
    "    model.add(Dropout(dropout_fraction))\n",
    "# Output layer\n",
    "    model.add(Dense(1))\n",
    "\n",
    "# Compile the model\n",
    "    model.compile(optimizer=\"adam\", loss=\"mean_squared_error\")\n",
    "\n",
    "# Train the model\n",
    "    model.fit(X_train, y_train, epochs=10, shuffle=False, batch_size=1, verbose=1)\n",
    "\n",
    "    return model\n",
    "\n",
    "\n",
    "\n",
    "\n",
    " "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "def lstm_prdiction(X_test):\n",
    "    y_predicted=model.predict(X_test)\n",
    "    prediction_df = pd.DataFrame({\n",
    "        \"Actual\":y_test.ravel(),\n",
    "        \"Predicted\":y_predicted.ravel(),\n",
    "    })\n",
    "\n",
    "    return prediction_df\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "def lstm_evaluation(X_test, y_test, y_predicted):\n",
    "    model_loss, model_accuracy = model.evaluate (X_test, y_test)\n",
    "    return model_loss, model_accuracy\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "def lstm_ploting(df):\n",
    "    lstm_plot = prediction_df.plot()\n",
    "    return lstm_plot\n",
    "\n",
    "    \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "interpreter": {
   "hash": "5b020d37e2d2792244d38dc749b48635155a26318d10fe99dbb63be9f4ff243c"
  },
  "kernelspec": {
   "display_name": "Python 3.9.7 ('base')",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  },
  "orig_nbformat": 4
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
