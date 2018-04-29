from portfolio import PortfolioGenerator
import pandas as pd
import numpy as np
from functools import reduce

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelBinarizer

import sys

class SampleStrategy(PortfolioGenerator):
    def __init__(self):
        # Read training set
        self.df = self.read_training_data()
        # Store features of our training set
        self.df = self.add_features(self.df)
        # Get models
        self.models = self.training_models()
        self.last_day_data = None
        self.last_day_features = None

    def build_signal(self, stock_features):
        return self.momentum(stock_features)

    def read_training_data(self):
        ticker_df = pd.read_csv('training_data/ticker_data.csv')
        factor_df = pd.read_csv('training_data/factor_data.csv')
        assert 'timestep' in ticker_df.columns, "ticker_data.csv has an invalid format"
        assert 'ticker' in ticker_df.columns, "ticker_data.csv has an invalid format"
        assert 'returns' in ticker_df.columns, "ticker_data.csv has an invalid format"
        assert 'timestep' in factor_df.columns, "factor_data.csv has an invalid format"
        ticker_df.set_index('timestep', inplace=True)
        factor_df.set_index('timestep', inplace=True)
        training_stock_df = ticker_df.join(factor_df, how='left')
        print('read_training_data_success\n')  # delete
        return training_stock_df

    def add_features(self, df):
        # Resort the data-frame to prepare for join
        df['timestep'] = df.index
        df.sort_values(by=['ticker', 'timestep'], inplace=True)
        df.reset_index(drop=True, inplace=True) # Reset index to natural numerical values
        # df:
        #         ticker timesteps
        #   0       0      0
        #   1       0      1
        #   2       0      2
        # .........

        # Generate fake price first
        fake_price = self.initial_fake_price(df)
        print('fake_price success')
        sys.stdout.flush()

        # Generate technical indicators
        MA1 = pd.Series(name='MA1')
        MA2 = pd.Series(name='MA2')
        MA3 = pd.Series(name='MA3')
        MOM = pd.Series(name='MOM')
        B1 = pd.Series(name='B1')
        B2 = pd.Series(name='B2')
        MACD = pd.Series(name='MACD')
        MACDsign = pd.Series(name='MACDsign')
        MACDdiff = pd.Series(name='MACDdiff')

        t = len(df.timestep.unique())
        for i in range(1000):
            # get fake price for a ticker
            prices = fake_price[i*t: (i+1)*t]
            # Calculate technical indicators
            ma1, ma2, ma3 = self.MA(prices)
            MA1 = MA1.append(ma1)
            MA2 = MA2.append(ma2)
            MA3 = MA3.append(ma3)
            MOM = MOM.append(self.MOM(prices))
            b1, b2 = self.BBANDS(prices)
            B1 = B1.append(b1)
            B2 = B2.append(b2)
            macd, macd_sign, macd_diff = self.MACD(prices)
            MACD = MACD.append(macd)
            MACDsign = MACDsign.append(macd_sign)
            MACDdiff = MACDdiff.append(macd_diff)
            print('Ticker ', i)

        # Form all information needed into a dataframe
        MA1 = MA1.to_frame('MA1')
        features = MA1.join(MA2.to_frame('MA2'), how='left').join(MA3.to_frame('MA3'),
        how='left').join(MOM.to_frame('MOM'), how='left').join(B1.to_frame('B1'),
        how='left').join(B2.to_frame('B2'), how='left').join(MACD.to_frame('MACD'),
        how='left').join(MACDsign.to_frame('MACDsign'), how='left').join(MACDdiff.to_frame('MACDdiff'),
        how='left').join(fake_price.to_frame('fake_price'), how='left')

        features = features.join(df['ticker'], how='left').join(df['timestep'],
        how='left').join(df['pb'],
        how='left').join(df['market_cap'], how='left').join(df['industry'],
        how='left').join(df['VIX'], how='left').join(df['3M_R'],
        how='left').join(df['BIG_IX'], how='left').join(df['SMALL_IX'],
        how='left').join(df['SENTI'], how='left').join(df['OIL'],
        how='left').join(df['returns'], how='left')

        # Quantify categorical data
        enc = LabelBinarizer()
        enc_results = enc.fit_transform(features['industry'])
        industry_binary = pd.DataFrame(enc_results, columns=enc.classes_)
        features = features.join(industry_binary, how='left')
        features = features.drop('industry', axis=1)

        # at the point, features should be sort by ticker and timestep, with natural index
        print("add_features success")
        return features

    def add_features_e(self, stock_features):
        # Generate fake price first
        last_1_prices = self.last_day_data[-1*1000:]['fake_price']
        fake_price = self.add_fake_price(stock_features, last_1_prices)

        # Join fake_price into stock_features
        stock_features['timestep'] = stock_features.index
        stock_features.reset_index(drop=True, inplace=True)
        stock_features.join(fake_price.to_frame('fake_price'), how='left')
        stock_features.set_index('timestep', drop=True, inplace=True)

        # get last 20 day data
        df = self.last_day_data[-20*1000:]
        df = df.append(stock_features)

        df.sort_values(by=['ticker', 'timestep'], inplace=True)
        fake_price = df['fake_price']


        # Generate technical indicators
        MA1 = pd.Series(name='MA1')
        MA2 = pd.Series(name='MA2')
        MA3 = pd.Series(name='MA3')
        MOM = pd.Series(name='MOM')
        B1 = pd.Series(name='B1')
        B2 = pd.Series(name='B2')
        MACD = pd.Series(name='MACD')
        MACDsign = pd.Series(name='MACDsign')
        MACDdiff = pd.Series(name='MACDdiff')

        t = len(fake_price.timestep.unique())
        for i in range(1000):
            # get fake price for a ticker
            prices = fake_price[i*t: (i+1)*t]
            # Calculate technical indicators
            ma1, ma2, ma3 = self.MA(prices)
            MA1 = MA1.append(ma1)
            MA2 = MA2.append(ma2)
            MA3 = MA3.append(ma3)
            MOM = MOM.append(self.MOM(prices))
            b1, b2 = self.BBANDS(prices)
            B1 = B1.append(b1)
            B2 = B2.append(b2)
            macd, macd_sign, macd_diff = self.MACD(prices)
            MACD = MACD.append(macd)
            MACDsign = MACDsign.append(macd_sign)
            MACDdiff = MACDdiff.append(macd_diff)
            print('Ticker ', i)

        # Form all information needed into a dataframe
        MA1 = MA1.to_frame('MA1')
        features = MA1.join(MA2.to_frame('MA2'), how='left').join(MA3.to_frame('MA3'),
        how='left').join(MOM.to_frame('MOM'), how='left').join(B1.to_frame('B1'),
        how='left').join(B2.to_frame('B2'), how='left').join(MACD.to_frame('MACD'),
        how='left').join(MACDsign.to_frame('MACDsign'), how='left').join(MACDdiff.to_frame('MACDdiff'),
        how='left').join(fake_price.to_frame('fake_price'), how='left')

        features = features.join(df['ticker'], how='left').join(df['timestep'],
        how='left').join(df['fake_price'], how='left').join(df['pb'],
        how='left').join(df['market_cap'], how='left').join(df['industry'],
        how='left').join(df['VIX'], how='left').join(df['3M_R'],
        how='left').join(df['BIG_IX'], how='left').join(df['SMALL_IX'],
        how='left').join(df['SENTI'], how='left').join(df['OIL'],
        how='left').join(df['returns'], how='left')

        features = features[-1000*50:]

        # Quantify categorical data
        enc = LabelBinarizer()
        enc_results = enc.fit_transform(features['industry'])
        industry_binary = pd.DataFrame(enc_results, columns=enc.classes_)
        features = features.join(industry_binary, how='left')
        features = features.drop('industry', axis=1)

        # at the point, features should be sort by ticker and timestep, with natural index
        print("add_features success")
        return features

    def training_models(self):
        print('Start learning and predicting')
        sys.stdout.flush()

        # self.df (features):
        #         ticker timesteps
        #   0       0      0
        #   1       0      1
        #   2       0      2
        # .........

        # Generate Models
        tscv = TimeSeriesSplit(n_splits=50)
        regressors = dict()
        t = len(self.df.timestep.unique())
        for i in range(1000):
            # Get features of a single stock
            ticker_data = self.df[i*t: (i+1)*t]
            # Delete all rows that contain NaN
            ticker_data = ticker_data.dropna()

            X = ticker_data.drop('returns', axis=1)
            y = ticker_data['returns']
            y.to_frame()
            X = X.values.reshape((len(X.index), len(X.columns)))
            y = y.values.reshape((len(y.index), 1))
            # y = np.nan_to_num(y)
            # X = np.nan_to_num(X)

            # Train Regressor
            reg = RandomForestRegressor()
            for train_index, test_index in tscv.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                rf = reg.fit(X_train, y_train.ravel())

            # Append trained regressor into the dictionary
            regressors[i] = reg
        return regressors

    def momentum(self, stock_features):
        # stock_features:
        #       tickers .....
        #   0      0    .....
        #   0      1    .....
        #   0      2    .....
        #  ..................
        signals = pd.Series(name='signal')
        timesteps = stock_features.index
        t_start = timesteps[0]
        t_end = timesteps[-1]
        t = len(stock_features.timestep.unique())

        # if the first timestep (0,49)
        if t_start == 0:
            my_features = self.add_features(stock_features)
            # my_features:
            #       tickers timestep .....
            #   0      0        0    .....
            #   1      0        1    .....
            #   2      0        2     .....
            #  ..................
        else:
            my_features = self.add_features_e(stock_features)

        # Use regressor to predict return at timestep 50
        t = len(my_features.timestep.unique())
        for i in range(1000):
            # Get features of a single stock
            X_test = my_features[i * t: (i + 1) * t]
            # Delete all rows that contain NaN
            X_test = X_test.dropna()
            signals[i] = self.models[i].predict(X_test)

        # Store data
        my_features.sort_values(by=['timestep', 'ticker'], inplace=True)
        my_features.set_index('timestep', drop=True, inplace=True)
        # my_features:
        #       ticker  .....
        # 0        0    .....
        # 0        1    .....
        # 0        2     .....
        #  ..................
        self.last_day_data = stock_features.join(my_features['fake_price'], how='left')
        return signals

    def initial_fake_price(self, df):
        #        ticker timesteps
        #   0       0      0
        #   1       0      1
        #   2       0      2
        # ............
        fake_price = np.zeros(len(df.index))
        returns = df['returns']
        t = len(df.timestep.unique())
        for i in range(len(df.index)):
            if i % t == 0:
                f = 100.0
            else:
                f = fake_price[i - 1] * (returns[i] + 1)
            fake_price[i] = f
        fake_price = pd.Series(fake_price, name='fake_price')
        print('fake_price success')
        sys.stdout.flush()
        return fake_price

    def add_fake_price(self, df, last_1_prices):
        # df:
        #  timesteps ticker  returns
        #   30       0          0.002
        #   30       1          0.023
        #   30       2          -0.012
        # .........

        fake_price = np.zeros(len(df.index))
        t_start = df.index[0]
        returns = df['returns']
        stock = 0
        for i in range(len(df.index)):
            # if the start day
            if df.index[i] == t_start:
                f = last_1_prices[stock]*(1 + returns[i])
                stock += 1
            else:
                f = fake_price[i-1000] * (returns[i] + 1)
            fake_price[i] = f
        fake_price = pd.Series(fake_price, name='fake_price')
        return fake_price

    # Moving Average
    def MA(self, prices, window1=5, window2=10, window3=20):
        MA1 = pd.Series(pd.rolling_mean(prices, window1), name='MA_' + str(window1))
        MA2 = pd.Series(pd.rolling_mean(prices, window2), name='MA_' + str(window2))
        MA3 = pd.Series(pd.rolling_mean(prices, window2), name='MA_' + str(window3))
        return MA1, MA2, MA3

    # Momentum
    def MOM(self, prices, window=9):
        M = pd.Series(prices.diff(window), name='Momentum_' + str(window))
        return M

    # Bollinger Bands
    def BBANDS(self, prices, window=20):
        MA = pd.Series(pd.rolling_mean(prices, window))
        MSD = pd.Series(pd.rolling_std(prices, window))
        b1 = 4 * MSD / MA
        B1 = pd.Series(b1, name='BollingerB_' + str(window))
        b2 = (prices - MA + 2 * MSD) / (4 * MSD)
        B2 = pd.Series(b2, name='Bollinger%b_' + str(window))
        return B1, B2

    # MACD, MACD Signal and MACD difference
    def MACD(self, prices, win_fast=26, win_slow=12):
        EMAfast = pd.Series(pd.ewma(prices, span=win_fast, min_periods=win_slow - 1))
        EMAslow = pd.Series(pd.ewma(prices, span=win_slow, min_periods=win_slow - 1))
        MACD = pd.Series(EMAfast - EMAslow, name='MACD_' + str(win_fast) + '_' + str(win_slow))
        MACDsign = pd.Series(pd.ewma(MACD, span=9, min_periods=8),
                             name='MACDsign_' + str(win_fast) + '_' + str(win_slow))
        MACDdiff = pd.Series(MACD - MACDsign, name='MACDdiff_' + str(win_fast) + '_' + str(win_slow))
        return MACD, MACDsign, MACDdiff

# Test out performance by running 'python sample_strategy.py'
if __name__ == "__main__":
    portfolio = SampleStrategy()
    sharpe = portfolio.simulate_portfolio()
    print("*** Strategy Sharpe is {} ***".format(sharpe))
