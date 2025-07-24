from backtrader_strategy_class import *
import os.path
import sys
import backtrader as bt
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime

import pandas as pd
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

START_DATE = datetime(2024, 1, 4)
END_DATE = datetime(2024, 1, 8)
TIMEFRAME_ALPACA = TimeFrame.Minute
TIMEFRAME_BT = bt.TimeFrame.Minutes
GET_DATA = True

if __name__ == '__main__':
    # Create a cerebro entity
    cerebro = bt.Cerebro()

    # Add a strategy
    strats = cerebro.optstrategy(
        OurStrategy,
        maperiod=range(180, 181))
    
    #cerebro.addstrategy(OurStrategy)

    # Datas are in a subfolder of the samples. Need to find where the script is
    # because it could have been called from anywhere
    modpath = os.path.dirname(os.path.abspath(sys.argv[0]))
    datapath = os.path.join(modpath, r'backtrade_data.csv')

    if GET_DATA:
        # Set API credentials
        API_KEY = "YOUR API KEY HERE"
        SECRET_KEY = "YOUR SECRET KEY HERE"

        # Initialize the Alpaca Data Client
        client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

        # Define the request parameters
        request_params = StockBarsRequest(
            symbol_or_symbols='SPY',
            timeframe=TIMEFRAME_ALPACA,
            start=START_DATE,
            end=END_DATE  # End date is exclusive
        )

        # Fetch the data
        print("Requesting data from Alpaca API")
        bars = client.get_stock_bars(request_params).df
        print("Data received")
        print("Saving data to CSV")
        bars.to_csv(datapath)
        print("Data saved to CSV")
        del bars

    # Create a Data Feed
    data = bt.feeds.GenericCSVData(
        dataname=datapath,
        fromdate=START_DATE,
        todate=END_DATE,
        timeframe=TIMEFRAME_BT,
        dtformat=("%Y-%m-%d %H:%M:%S%z"),
        datetime=1,
        open=2,
        high=3,
        low=4,
        close=5,
        volume=6,
        )

    # Add the Data Feed to Cerebro
    cerebro.adddata(data)

    # Set our desired cash start
    cerebro.broker.setcash(1000.0)

    # Add a FixedSize sizer according to the stake
    cerebro.addsizer(bt.sizers.FixedSize, stake=1)

    # Set the commission
    cerebro.broker.setcommission(commission=0.0)

    # Run over everything
    cerebro.run(maxcpus=None)
