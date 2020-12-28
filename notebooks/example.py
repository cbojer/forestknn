# %%
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from forestknn import *

path = "~/Documents/kaggle/walmart/"

# %%
data = (
    pd.read_csv(path + "train.csv", parse_dates=True)
    .assign(Date=lambda x: pd.to_datetime(x["Date"]))
    .assign(month=lambda x: x["Date"].dt.month)
    .assign(day=lambda x: x["Date"].dt.day)
    .assign(dayofyear=lambda x: x["Date"].dt.dayofyear)
)
x = data.iloc[0:10000].drop(["Weekly_Sales", "Date"], axis=1)
y = data.iloc[0:10000].loc[:, "Weekly_Sales"]

#%%
rf = RandomForestRegressor()
rf.fit(x, y)

#%%
q = data.iloc[100100:100101, :].drop(["Weekly_Sales", "Date"], axis=1)
q
#%%
knn = adaptive_k_nearest_neighbors(rf, x, q)
knn

#%%
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

imp = permutation_importance(rf, x, y)
plt.bar(x.columns, imp["importances_mean"])