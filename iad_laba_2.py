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

"""Присваиваем x y к датасету"""
x, y = make_friedman2(random_state=0, n_samples=10000, noise=0.5)
"""Создаем тренеровочные данные и тестовые"""
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
"""Ищем лучшие гиперпараметры"""
pipe_linearsvr = Pipeline([('scaler', StandardScaler()), ('linear_svr', LinearSVR())])
pipe_svr_linear = Pipeline([('scaler', StandardScaler()), ('linear_svr', SVR(kernel='linear'))])
pipe_svr_poly = Pipeline([('scaler', StandardScaler()), ('linear_svr', SVR(kernel='poly'))])
pipe_svr_rbf = Pipeline([('scaler', StandardScaler()), ('linear_svr', SVR(kernel='rbf'))])

search_linearsvr = GridSearchCV(pipe_linearsvr, {'linear_svr__epsilon': [0.1, 0.2, 0.3, 0.5]})
search_svr_linear = GridSearchCV(pipe_svr_linear, {'linear_svr__epsilon': [0.1, 0.2, 0.3, 0.5]})
search_svr_poly = GridSearchCV(pipe_svr_poly, {'linear_svr__epsilon': [0.1, 0.2, 0.3, 0.5],
                                               'linear_svr__C': [0.1, 0.5, 1, 2],
                                               'linear_svr__degree': [1, 2, 3, 4, 5, 6, 7]})
search_svr_rbf = GridSearchCV(pipe_svr_rbf, {'linear_svr__gamma': [0.1, 0.2, 0.3, 0.5],
                                             'linear_svr__C': [0.1, 0.5, 1, 2]})

search_linearsvr.fit(x, y)
search_svr_linear.fit(x, y)
search_svr_poly.fit(x, y)
search_svr_rbf.fit(x, y)

print(search_linearsvr.best_params_)
print(search_svr_linear.best_params_)
print(search_svr_poly.best_params_)
print(search_svr_rbf.best_params_)


pd_metrics_search = pd.DataFrame(data=[[search_linearsvr.score(x_test, y_test), search_svr_linear.score(x_test, y_test),
                                        search_svr_poly.score(x_test, y_test), search_svr_rbf.score(x_test, y_test)],
                                       [mean_squared_error(y_test, search_linearsvr.predict(x_test)), mean_squared_error(
                                           y_test, search_svr_linear.predict(x_test)),
                                        mean_squared_error(y_test, search_svr_poly.predict(x_test)), mean_squared_error(
                                           y_test, search_svr_rbf.predict(x_test))],
                                       [mean_absolute_error(y_test, search_linearsvr.predict(x_test)), mean_absolute_error(
                                           y_test, search_svr_linear.predict(x_test)),
                                        mean_absolute_error(y_test, search_svr_poly.predict(x_test)), mean_absolute_error(
                                           y_test, search_svr_rbf.predict(x_test))],
                                       [mean_absolute_percentage_error(y_test, search_linearsvr.predict(x_test)),
                                        mean_absolute_percentage_error(y_test, search_svr_linear.predict(x_test)),
                                        mean_absolute_percentage_error(y_test, search_svr_poly.predict(x_test)),
                                        mean_absolute_percentage_error(y_test, search_svr_rbf.predict(x_test))]],
                                 index=['R2', 'RMSE', 'MAE', "MAPE"],
                                 columns=["LinearSVR", "SVR_Linear", "SVR_poly", "SVR_rbf"])
print(pd_metrics_search)
print('//'*100)

fig1, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 10))

ax1.scatter(y_test, search_linearsvr.predict(x_test))
ax1.set_title('LinearSVR')
ax2.scatter(y_test, search_svr_linear.predict(x_test))
ax2.set_title('SVR linear')
ax3.scatter(y_test, search_svr_poly.predict(x_test))
ax3.set_title('SVR poly')
ax4.scatter(y_test, search_svr_rbf.predict(x_test))
ax4.set_title('SVR rbf')

plt.show()
