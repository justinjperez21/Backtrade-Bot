import backtrader as bt
#import torch
import pandas as pd
import statsmodels.api as sm
from random import random
import numpy as np
from strategy_logic import *
from torch.utils.tensorboard import SummaryWriter

class OurStrategy(bt.Strategy):
    params = (
        ('maperiod', 30),
        ('printlog', True),
    )

    def log(self, txt, dt=None, doprint=False):
        ''' Logging function fot this strategy'''
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close

        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.net_profit = 0
        self.num_trades = 0
        self.cur_next_step = 0
        self.prev_preds = []
        self.start_price = 0#self.data.close[0]

        self.writer = SummaryWriter(log_dir="bt_runs/experiment5")

        # Add a MovingAverageSimple indicator
        self.sma = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.maperiod)

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        '''if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')'''

        # Write down: no pending order
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.net_profit += trade.pnl
        self.num_trades += 1
        self.profit_per_trade = self.net_profit / self.num_trades
        self.market_profit = (self.data.close[0] - self.start_price) / self.start_price

        self.writer.add_scalar("Profits/Gross", trade.pnl, self.cur_next_step)
        self.writer.add_scalar("Profits/Net", self.net_profit, self.cur_next_step)
        self.writer.add_scalar("Profits/Per Trade", self.profit_per_trade, self.cur_next_step)

        #self.cur_trade_step += 1

        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f, PER_TRADE %.5f, MARKET %.5f' %
                 (trade.pnl, self.net_profit, self.profit_per_trade, self.market_profit))

    def get_data_df(self):
        return pd.DataFrame({
            'close': self.data.close.get(size=self.params.maperiod),
            'low': self.data.low.get(size=self.params.maperiod),
            'high': self.data.high.get(size=self.params.maperiod),
            'open': self.data.open.get(size=self.params.maperiod),
            'volume': self.data.volume.get(size=self.params.maperiod),
        })

    def next(self):
        if self.start_price == 0:
            self.start_price = self.data.close[0]

        self.writer.add_scalar("Data/Close", self.data.close[0], self.cur_next_step)
        
        df = self.get_data_df()
        if len(self.prev_preds) >= len(df):
            df['prev_preds'] = self.prev_preds[-len(df):]
        pred = RFStrategy(df, self.writer, self.cur_next_step)
        self.writer.add_scalar("Profits/Prediction", pred, self.cur_next_step)
        self.prev_preds.append(pred)
        self.prev_preds = self.prev_preds[-len(df):]
        self.cur_next_step += 1

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # Simply log the closing price of the series from the reference
        #self.log('Close, %.2f' % self.dataclose[0])
        
        # Check if we are in the market
        if not self.position:

            # Not yet ... we MIGHT BUY if ...
            if pred > 0:

                # BUY, BUY, BUY!!! (with all possible default parameters)
                #self.log('BUY CREATE, %.2f' % self.dataclose[0])

                # Keep track of the created order to avoid a 2nd order
                self.order = self.buy()
                #self.recently_bought = True

        else:

            if pred < 0:
                # SELL, SELL, SELL!!! (with all possible default parameters)
                #self.log('SELL CREATE, %.2f' % self.dataclose[0])

                # Keep track of the created order to avoid a 2nd order
                self.order = self.sell()

    def stop(self):
        self.log('(MA Period %2d) Ending Value %.2f' %
                 (self.params.maperiod, self.broker.getvalue()), doprint=True)