import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_friedman2
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVR, SVR
from sklearn.model_selection import GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt
import time

g = time.time()
"""Присваиваем x y к датасету"""
x, y = make_friedman2(random_state=0, n_samples=1000, noise=0.5)
"""Создаем тренеровочные данные и тестовые"""
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
"""Ищем лучшие гиперпараметры"""

g1 = time.time()

best_est1 = Pipeline([('scaler', StandardScaler()), ('svr', LinearSVR(epsilon=0.02))])
best_est2 = Pipeline([('scaler', StandardScaler()), ('svr', SVR(kernel='linear', epsilon=0.07))])
best_est3 = Pipeline([('scaler', StandardScaler()), ('svr', SVR(kernel='poly', C=3, epsilon=0.09, degree=3))])
best_est4 = Pipeline([('scaler', StandardScaler()), ('svr', SVR(kernel='rbf', C=3, gamma=0.07))])

best_est1.fit(x_train, y_train)
best_est2.fit(x_train, y_train)
best_est3.fit(x_train, y_train)
best_est4.fit(x_train, y_train)

best_est_predict1 = best_est1.predict(x_test)
best_est_predict2 = best_est2.predict(x_test)
best_est_predict3 = best_est3.predict(x_test)
best_est_predict4 = best_est4.predict(x_test)

pd_metrics_search = pd.DataFrame(data=[[best_est1.score(x_test, y_test), best_est2.score(x_test, y_test),
                                        best_est3.score(x_test, y_test), best_est4.score(x_test, y_test)],
                                       [mean_squared_error(y_test, best_est_predict1),
                                        mean_squared_error(y_test, best_est_predict2),
                                        mean_squared_error(y_test, best_est_predict3),
                                        mean_squared_error(y_test, best_est_predict4)],
                                       [mean_absolute_error(y_test, best_est_predict1),
                                        mean_absolute_error(y_test, best_est_predict2),
                                        mean_absolute_error(y_test, best_est_predict3),
                                        mean_absolute_error(y_test, best_est_predict4)],
                                       [mean_absolute_percentage_error(y_test, best_est_predict1),
                                        mean_absolute_percentage_error(y_test, best_est_predict2),
                                        mean_absolute_percentage_error(y_test, best_est_predict3),
                                        mean_absolute_percentage_error(y_test, best_est_predict4)]],
                                 index=['R2', 'RMSE', 'MAE', "MAPE"],
                                 columns=["LinearSVR", "SVR_Linear", "SVR_poly", "SVR_rbf"])
print(pd_metrics_search)
# pd_metrics_search.to_csv('out.csv')
print('//' * 100)
print(time.time() - g)
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 10))
ax1.plot(best_est_predict1, color="red", lw=2)
ax1.scatter(y_test, best_est_predict1)
ax1.set(xlabel='right y', ylabel='predict y')
ax1.set_title('LinearSVR')
ax2.scatter(y_test, best_est_predict2)
ax2.set(xlabel='right y', ylabel='predict y')
ax2.set_title('SVR linear')
ax3.scatter(y_test, best_est_predict3)
ax3.set(xlabel='right y', ylabel='predict y')
ax3.set_title('SVR poly')
ax4.scatter(y_test, best_est_predict4)
ax4.set(xlabel='right y', ylabel='predict y')
ax4.set_title('SVR rbf')
plt.show()
# plt.savefig('saved_figure.png')
