# Project 2 Proposal

* Interests and Intent
1. We're going to focus on the markets and build an algorithmic trading bot.

2. We want to predict a whether or not an ETF will stay within a a price range over a certain period of time.

3. The strategy will be used to identify times when the market has underpriced or overpriced volatility. The trading bot will then signal a short iron condor or strangle with short strike prices. 


* Data sources and ML models
1. We will use Alpaca API to gather SPY ETF historical data, and VIX data from 2015-2022.

2. For technical indicators, we will use the finta library
    Range Indicators:
    -Average True Range (ATR)
    -Keltner Channels(KC)
    -Mass Index (MI)
    
    Momentum indicators:
    -Volatility Based Momentum(VBM)
    - Dynamic Momentum Index(DYMI)
    -Adaptive Price Zone (APZ)

    Moving Average indicators:
    -BBands
    -KAMA


3. Model for Range Prediction:
---
    Long Short Term Memory (LSTM)- uses historical data to predict future values. Good for time series data.


Project Outline:
---
    Data Gathering: 
        a) OHLCV data for
            SPY
    Data Cleaning:
        a) read into dfs 
        b) add indicators 
    Data Preprocessing:
        a) feature - target separation
        b) test and training data scaling
        c) reshaping
    Model Testing
    Model Evaluation
    Model Prediction
    Visualizations


ETFs:
---
 Market Index: SPY 

