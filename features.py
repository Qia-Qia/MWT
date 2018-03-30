from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

MAX_LOOKBACK = 50

select = RFE(RandomForestRegressor(n_estimators=100, random_state=42),
                 n_features_to_select=10)
ticker_df = pd.read_csv('stock_data/ticker_data.csv')
factor_df = pd.read_csv('stock_data/factor_data.csv')
ticker_df.set_index('timestep', inplace=True)
factor_df.set_index('timestep', inplace=True)
stock_df = ticker_df.join(factor_df, how='left')

for idx in stock_df.index.unique():
    if idx < MAX_LOOKBACK:
        continue
    stock_features = stock_df.loc[idx-MAX_LOOKBACK:idx-1]
    returns = stock_df.loc[idx:idx].set_index('ticker')['returns']

returns.values.reshape((1000,1))
X_train, X_test, y_train, y_test = train_test_split(stock_features, returns, random_state = 0)
select.fit(X_train, y_train)
# visualize the selected features:
mask = select.get_support()
plt.matshow(mask.reshape(1, -1), cmap='gray_r')
plt.xlabel("Sample index")
plt.yticks(())
