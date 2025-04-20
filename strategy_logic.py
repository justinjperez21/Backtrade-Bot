import pandas as pd
import statsmodels.api as sm
from random import random
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
#from xgboost import XGBRegressor

def get_lagged_df(df, context_length, drop_cols=[]):
    Xy = pd.DataFrame()

    # Create a column for the target variable
    Xy['y'] = df['open']#.shift(-1)
    
    # Create a column for the lagged values
    for col in df.columns:
        if col not in drop_cols:
            for i in range(1, context_length+1):
                Xy[f'{col}_{i}'] = df[col].shift(i)

    for col in drop_cols:
        Xy[col] = df[col]

    # Drop rows with NaN values
    Xy = Xy.dropna().reset_index(drop=True)

    return Xy

'''
def OLSStrategy(df):
    rank = 0
    for col in df.columns:
        for p in range(1, rank+1):
            df[col + f'_exp_{p}'] = df[col]**p
    Xy = get_lagged_df(df, 1)#self.params.maperiod//2)
    Xy = Xy.copy()
    Xy['constant'] = 1
    Xy['y'] = Xy['y'] - Xy['high_1']
    #y = df['y'].diff().iloc[1:].reset_index(drop=True)
    curr_row = pd.DataFrame()
    for col in df.columns:
        for i in range(0, 1):#self.params.maperiod//2):
            curr_row[f'{col}_{i+1}'] = df[col].shift(i)
    curr_row['constant'] = 1
    curr_row = curr_row.iloc[-1]
    #curr_row = df.iloc[-1]
    #df = df.iloc[1:].reset_index(drop=True)
    model = sm.OLS(Xy['y'], Xy.drop(columns='y'))
    results = model.fit()
    pred = float(results.predict(curr_row))
    #from IPython import embed; embed()
    
    mean_diff = Xy['y'].abs().mean()
    std_diff = Xy['y'].std()
    allowable = 0

    if pred > allowable:
        buy_or_sell = "buy"
    elif pred < -allowable:
        buy_or_sell = "sell"
    else:
        buy_or_sell = "hold"
    return buy_or_sell
'''
    
def RFStrategy(df, writer, cur_step):
    df['y'] = df['close'].shift(-1) - df['close']
    horizons = [5, 20, 30, 60]
    if 'prev_preds' in df.columns:
        df['error'] = df['close'].diff() - df['prev_preds']
        #df = df.drop(columns='prev_preds')
    for horizon in horizons:
        #rol_sum = df.rolling(window=horizon).sum()
        rol_avg = df.rolling(window=horizon).mean()
        rol_std = df.rolling(window=horizon).std()

        df[f'close_avg_ratio_{horizon}'] = df['close'] / (rol_avg['close'])
        df[f'close_std_ratio_{horizon}'] = df['close'] / (rol_std['close'] + 1e-5)
        df[f'HLR_{horizon}'] = rol_avg['high'] / rol_avg['low']
        df[f'trend_{horizon}'] = df.shift(1).rolling(window=horizon).mean()['y']

        if 'prev_preds' in df.columns:
            df[f'error_mean_{horizon}'] = rol_avg['error']
            df[f'prev_pred_mean_{horizon}'] = rol_avg['prev_preds']

    if 'prev_preds' in df.columns:
        df = df.drop(columns=['prev_preds', 'error'])
    drop_cols = ['close', 'open', 'high', 'low']
    df = df.drop(columns=drop_cols)
    curr_row = df.iloc[-1].to_frame().T.drop(columns='y')
    df = df.iloc[:-1].dropna().reset_index(drop=True)
    model = RandomForestRegressor(
                #n_estimators=100,
                #colsample_bytree=0.3,
                #learning_rate=0.3,
                #max_depth=3,
                #n_jobs=-1, # Can't use this with cerebro multi-processing
                random_state=0,
              )
    model.fit(
                df.drop(columns='y'),
                df['y'],
                #sample_weight=df.index**(1/len(df)**0.5),
              )
    for i in range(len(model.feature_importances_)):
        writer.add_scalar(f"Feature_Importance/{df.drop(columns='y').columns[i]}", model.feature_importances_[i], cur_step)
    pred = model.predict(curr_row)[0]
    
    mean_diff = df['y'].abs().mean()
    std_diff = df['y'].std()
    #if 'error' in curr_row.columns:
    #    allowable = curr_row['error']
    #allowable = 0

    #if pred > allowable:
    #    pred = pred
    #elif pred < -allowable:
    #    pred = pred
    #else:
    #    pred = 0
    return pred