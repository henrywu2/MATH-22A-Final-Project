import numpy as np
# import pandas as pd
from sklearn.linear_model import LinearRegression

X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
# y = 1 * x_0 + 2 * x_1 + 3
y = np.dot(X, np.array([1, 2])) + 3
reg = LinearRegression().fit(X, y)
print(reg.score(X, y))

print(reg.coef_)

print(reg.intercept_)

print(reg.predict(np.array([[3, 5]])))

def cumulative_analysis(database='fatal_encounters'):
  if database == 'fatal_encounters':
    df = pd.read_csv('https://docs.google.com/spreadsheets/d/1dKmaV_JiWcG8XBoRgP8b4e9Eopkpgt7FL7nyspvzAsE/export?format=csv&gid=0').rename(columns={'Date of injury resulting in death (month/day/year)':'date'}).iloc[:-1, :]
  elif database == 'wapo':
    df = pd.read_csv('https://github.com/washingtonpost/data-police-shootings/releases/download/v0.1/fatal-police-shootings-data.csv')
  df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
  #df = df.loc[df['Race with imputations'] == 'African-American/Black']
  df = df.loc[df['race'] == 'B']
  df = cumulative(df, 'date', date(2015, 1, 1))
  for i in df.index:
    df.at[i, 'year'] = i.year
  df = df.groupby(['year']).sum()
  print(df)