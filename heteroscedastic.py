import numpy as np
import pandas as pd
import statsmodels.api as sm
pd.set_option('display.max_rows', 6000)
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

income = pd.read_csv('intrvw20/itbi204.csv')
income = income.loc[income['UCC'] == 980071]
income = income.groupby(income['NEWID']).sum()
income = income[['UCC', 'VALUE']]
# print(income)

food = pd.read_csv('intrvw20/fmli204.csv')
food = food.set_index('NEWID')
food = food[['FOODPQ', 'ALCBEVPQ', 'APPARPQ', 'FOODCQ', 'ALCBEVCQ', 'APPARCQ']]
# print(food)

df = income.merge(food, how = 'inner', left_index = True, right_index = True)
df = df.sort_values(by = 'VALUE')
print(df)

stat = 'FOODPQ'

df_value = df['VALUE'].to_numpy().reshape(-1, 1)
df_stat = df[stat].to_numpy().reshape(-1, 1)

# ordinary linear regression
reg = LinearRegression().fit(df_value, df_stat)
score = reg.score(df_value, df_stat)
pred = reg.predict(df_value)
print(f'ORDINARY | Score: {score}, Coefficient: {reg.coef_[0][0]}, Intercept: {reg.intercept_[0]}')
resid = df_stat - pred

df_value = sm.add_constant(df_value)

# ordinary linear regression with other package (check)
# oreg = sm.OLS(df_stat, df_value)
# oresults = oreg.fit()
# print(f'ORDINARY | Score: {oresults.rsquared}, Coefficient: {oresults.params[1]}, Intercept: {oresults.params[0]}')

# weighted linear regression
CUTOFF = 20000
df_low = df.loc[income['VALUE'] < CUTOFF]
df_high = df.loc[income['VALUE'] >= CUTOFF]
mse_low = mean_squared_error(df_low['VALUE'], pred[:len(df_low.index)])
mse_high = mean_squared_error(df_high['VALUE'], pred[len(df_low.index):])

weights = np.append(np.repeat(1 / mse_low, len(df_low.index)), np.repeat(1 / mse_high, len(df_high.index)))
wreg = sm.WLS(df_stat, df_value, weights = weights)
wresults = wreg.fit()
wpred = wresults.predict()
print(f'WEIGHTED | Score: {wresults.rsquared}, Coefficient: {wresults.params[1]}, Intercept: {wresults.params[0]}')

plt.scatter(df['VALUE'], df[stat], s=1, c = 'k', label = 'Responses')
plt.plot(df['VALUE'], pred, c = 'b', label = 'OLS (' + r'y = 861.573 + 0.032x' + ')')
plt.plot(df['VALUE'], wpred, c = 'g', label = 'WLS (' + r'y = 805.793 + 0.036x' + ')')
plt.gca().axhline(y = 0, color = '0.5', zorder = 0)
plt.gca().axvline(x = 0, color = '0.5', zorder = 0)
plt.xlabel('Income (USD)', size = 16)
plt.ylabel('Spending on Food (USD)', size = 16)
plt.title('Income and Expenditures Over a Three-Month Period in 2020', size = 24)
plt.legend()
plt.show()

plt.scatter(df['VALUE'], resid, s=1, c = 'k')
plt.ylabel('Residuals of Spending on Food (USD)', size = 16)
plt.xlabel('Income (USD)', size = 16)
plt.gca().axhline(y = 0, color = '0.5', zorder = 0)
plt.gca().axvline(x = 0, color = '0.5', zorder = 0)
plt.xlim(-1000, 100000)
plt.ylim(top = 4000)
plt.title('Income and Food Expenditures (Residuals) Over a Three-Month Period in 2020', size = 24)
plt.show()

# plt.hist(df['VALUE'], bins=2000)
# plt.show()