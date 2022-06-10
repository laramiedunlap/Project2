# Project 2 Proposal

* Interests and Intent
1. We're going to focus on the markets and build an algorithmic trading bot.

2. We want to predict a whether or not an ETF will stay within a a price range over a certain period of time.

3. The strategy will be used to identify times when the market has underpriced or overpriced volatility. The trading bot will then signal a short iron condor or strangle with short strike prices. 


* Data sources and ML models
1. We will use Alpaca API to gather ETF historical data, 2015-2022.

2. For technical indicators, we will use the finta library

3. Models for Range Prediction:
---
    ANN
    LSTM

4. Models for Classifing in-range and out-of-range trades:
---
    Random Forest Classifier
    RNN
    K-means Clustering


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
    Model Testing
    Model Evaluation
    Model Prediction
    Visualizations


    * Make a .ipynb with your name
    * feel free to make .py files with utilities

ETFs:
---
    Emerging Markets ETF: EWM
    Russell (small caps): IWM
    ** Market Index: SPY **
    Fincance, energy: XLF , XLE
    Cons. Disc.: XLY
    Gold Miners: GDX
    Tech: XLK
    SemiCond.: SMH