#%%
import numpy as np 
import pandas as pd


# question 1
pd.__version__ 
#%%
# question 2
file_url = "https://raw.githubusercontent.com/alexeygrigorev/datasets/master/housing.csv"

df = pd.read_csv(file_url)
df.shape[1]
#%%
# question 3
null_columns=df.notnull().all(axis=0)
mask=null_columns==False
null_columns.index[mask]
#%%
# question 4
len(df.ocean_proximity.unique())
#%%
# question 5
df[df.ocean_proximity=='NEAR BAY'].median_house_value.mean()
#%%
# question 6
mean_before_fillna=df.total_bedrooms.mean()
# mean_after_fillna=df.total_bedrooms.ffill().mean() # wrong! the question instructs to fillna with mean!
mean_after_fillna=df.total_bedrooms.fillna(df.total_bedrooms.mean()).mean()
print(f"mean before fillna = {mean_before_fillna}\nmean after fillna = {mean_after_fillna}")
#%%
# question 7
df_island=df[df.ocean_proximity=='ISLAND']
df_island[['housing_median_age']]
df_island=df[df.ocean_proximity=='ISLAND']
df_island=df_island[['housing_median_age','total_rooms','total_bedrooms']]
x=df_island.to_numpy()
xtx = x.T@x
xtx_inv = np.linalg.inv(xtx)
y=np.array([950, 1300, 800, 1000, 1300])
w=(xtx_inv@x.T)@y
w[-1]
