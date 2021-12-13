import numpy as np
import pandas as pd
import statsmodels.api as sm
pd.set_option('display.max_rows', 30)
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

launches = np.array([[53, 3], [75, 2], [57, 1], [58, 1], [63, 1], [70, 1], [70, 1], [66, 0], [67, 0], [67, 0], [67, 0], [68, 0], [69, 0], [70, 0], [70, 0], [72, 0], [73, 0], [75, 0], [76, 0], [76, 0], [78, 0], [79, 0], [80, 0], [81, 0]])
launches = pd.DataFrame(launches, columns = ['temp', 'failures'])
print(launches)

launches_temps = launches['temp'].to_numpy().reshape(-1, 1)
launches_failures = launches['failures'].to_numpy().reshape(-1, 1)

# ordinary linear regression
reg = LinearRegression().fit(launches_temps, launches_failures)
score = reg.score(launches_temps, launches_failures)
pred = reg.predict(launches_temps)
print(f'ORDINARY | Score: {score}, Coefficient: {reg.coef_[0][0]}, Intercept: {reg.intercept_[0]}')

launches_temps = sm.add_constant(launches_temps)

# ordinary linear regression with other package (check)
# oreg = sm.OLS(launches_failures, launches_temps)
# oresults = oreg.fit()
# print(f'ORDINARY | Score: {oresults.rsquared}, Coefficient: {oresults.params[1]}, Intercept: {oresults.params[0]}')

# weighted linear regression
weights = 1 / (launches['temp'] - 31)
wreg = sm.WLS(launches_failures, launches_temps, weights = weights)
wresults = wreg.fit()
wpred = wresults.predict()
print(f'WEIGHTED | Score: {wresults.rsquared}, Coefficient: {wresults.params[1]}, Intercept: {wresults.params[0]}')

plt.scatter(launches['temp'], launches['failures'], s = 20, color = 'k', label = 'Past shuttle launches')

plt.gca().axhline(y = 0, c = '0.5', zorder = 0)
plt.plot([0, 100], [reg.intercept_[0], reg.coef_[0][0] * 100 + reg.intercept_[0]], color = 'b', label = 'OLS (' + r'y = 4.675 - 0.061x' + ')')
plt.scatter(31, reg.coef_[0][0] * 31 + reg.intercept_[0], color = 'b', label = 'OLS prediction')
plt.annotate('(31, 2.789)', (31, 2.789), (-100, -30), textcoords = 'offset pixels')

plt.plot([0, 100],  [wresults.params[0], wresults.params[1] * 100 + wresults.params[0]], color = 'g', label = 'OLS (' + r'y = 5.601 - 0.074x' + ')')
plt.scatter(31, wresults.params[1] * 31 + wresults.params[0], color = 'g', label = 'WLS prediction')
plt.annotate('(31, 3.305)', (31, 3.305), (5, 5), textcoords = 'offset pixels')

plt.xlabel('Temperature (°F)', size = 16)
plt.ylabel('O-Ring Failures', size = 16)
plt.xlim(20, 90)
plt.ylim(-0.4, 5)
plt.title('Shuttle Launches', size = 24)
plt.legend()
plt.show()