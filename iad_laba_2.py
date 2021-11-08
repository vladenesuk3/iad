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
x, y = make_friedman2(random_state=0, n_samples=15000, noise=0.5)
"""Создаем тренеровочные данные и тестовые"""
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
"""Ищем лучшие гиперпараметры"""
pipe_linearsvr = Pipeline([('scaler', StandardScaler()), ('linear_svr', LinearSVR())])
pipe_svr_linear = Pipeline([('scaler', StandardScaler()), ('linear_svr', SVR(kernel='linear'))])
pipe_svr_poly = Pipeline([('scaler', StandardScaler()), ('linear_svr', SVR(kernel='poly'))])
pipe_svr_rbf = Pipeline([('scaler', StandardScaler()), ('linear_svr', SVR(kernel='rbf'))])

search_linearsvr = GridSearchCV(pipe_linearsvr, {'linear_svr__epsilon': np.arange(0.001, 0.25, 0.005)})
search_svr_linear = GridSearchCV(pipe_svr_linear, {'linear_svr__epsilon': np.arange(0.001, 0.25, 0.005)})
search_svr_poly = GridSearchCV(pipe_svr_poly, {'linear_svr__epsilon': np.arange(0.001, 0.25, 0.001),
                                               'linear_svr__C': np.arange(0.5, 10, 0.5),
                                               'linear_svr__degree': [2, 3, 4, 5]})
search_svr_rbf = GridSearchCV(pipe_svr_rbf, {'linear_svr__gamma': np.arange(0.001, 0.025, 0.005),
                                             'linear_svr__C': np.arange(0.5, 10, 0.5)})

search_linearsvr.fit(x_train, y_train)
search_svr_linear.fit(x_train, y_train)
search_svr_poly.fit(x_train, y_train)
search_svr_rbf.fit(x_train, y_train)

print(search_linearsvr.best_params_)
print(search_svr_linear.best_params_)
print(search_svr_poly.best_params_)
print(search_svr_rbf.best_params_)

pd_metrics_search = pd.DataFrame(data=[[search_linearsvr.score(x_test, y_test), search_svr_linear.score(x_test, y_test),
                                        search_svr_poly.score(x_test, y_test), search_svr_rbf.score(x_test, y_test)],
                                       [mean_squared_error(y_test, search_linearsvr.predict(x_test)),
                                        mean_squared_error(y_test, search_svr_linear.predict(x_test)),
                                        mean_squared_error(y_test, search_svr_poly.predict(x_test)),
                                        mean_squared_error(y_test, search_svr_rbf.predict(x_test))],
                                       [mean_absolute_error(y_test, search_linearsvr.predict(x_test)),
                                        mean_absolute_error(y_test, search_svr_linear.predict(x_test)),
                                        mean_absolute_error(y_test, search_svr_poly.predict(x_test)),
                                        mean_absolute_error(y_test, search_svr_rbf.predict(x_test))],
                                       [mean_absolute_percentage_error(y_test, search_linearsvr.predict(x_test)),
                                        mean_absolute_percentage_error(y_test, search_svr_linear.predict(x_test)),
                                        mean_absolute_percentage_error(y_test, search_svr_poly.predict(x_test)),
                                        mean_absolute_percentage_error(y_test, search_svr_rbf.predict(x_test))]],
                                 index=['R2', 'RMSE', 'MAE', "MAPE"],
                                 columns=["LinearSVR", "SVR_Linear", "SVR_poly", "SVR_rbf"])
print(pd_metrics_search)
pd_metrics_search.to_csv('out.csv')
print('//' * 100)
print(time.time() - g)
fig1, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 10))

ax1.scatter(y_test, search_linearsvr.predict(x_test))
ax1.set_title('LinearSVR')
ax2.scatter(y_test, search_svr_linear.predict(x_test))
ax2.set_title('SVR linear')
ax3.scatter(y_test, search_svr_poly.predict(x_test))
ax3.set_title('SVR poly')
ax4.scatter(y_test, search_svr_rbf.predict(x_test))
ax4.set_title('SVR rbf')

# plt.show()
plt.savefig('saved_figure.png')

pipe_linearsvr1 = Pipeline([('scaler', StandardScaler()), ('linear_svr', LinearSVR())])
pipe_svr_linear1 = Pipeline([('scaler', StandardScaler()), ('linear_svr', SVR(kernel='linear'))])
pipe_svr_poly1 = Pipeline([('scaler', StandardScaler()), ('linear_svr', SVR(kernel='poly'))])
pipe_svr_rbf1 = Pipeline([('scaler', StandardScaler()), ('linear_svr', SVR(kernel='rbf'))])

search_linearsvr1 = GridSearchCV(pipe_linearsvr1, {'linear_svr__epsilon': np.arange(0.25, 0.5, 0.005)})
search_svr_linear1 = GridSearchCV(pipe_svr_linear1, {'linear_svr__epsilon': np.arange(0.25, 0.5, 0.005)})
search_svr_poly1 = GridSearchCV(pipe_svr_poly1, {'linear_svr__epsilon': np.arange(0.25, 0.5, 0.001),
                                                 'linear_svr__C': np.arange(0.5, 10, 0.5),
                                                 'linear_svr__degree': [2, 3, 4, 5]})
search_svr_rbf1 = GridSearchCV(pipe_svr_rbf1, {'linear_svr__gamma': np.arange(0.25, 0.5, 0.005),
                                               'linear_svr__C': np.arange(0.5, 10, 0.5)})

search_linearsvr1.fit(x_train, y_train)
search_svr_linear1.fit(x_train, y_train)
search_svr_poly1.fit(x_train, y_train)
search_svr_rbf1.fit(x_train, y_train)

print(search_linearsvr1.best_params_)
print(search_svr_linear1.best_params_)
print(search_svr_poly1.best_params_)
print(search_svr_rbf1.best_params_)

pd_metrics_search1 = pd.DataFrame(
    data=[[search_linearsvr.score(x_test, y_test), search_svr_linear.score(x_test, y_test),
           search_svr_poly.score(x_test, y_test), search_svr_rbf.score(x_test, y_test)],
          [mean_squared_error(y_test, search_linearsvr.predict(x_test)),
           mean_squared_error(y_test, search_svr_linear.predict(x_test)),
           mean_squared_error(y_test, search_svr_poly.predict(x_test)),
           mean_squared_error(y_test, search_svr_rbf.predict(x_test))],
          [mean_absolute_error(y_test, search_linearsvr.predict(x_test)),
           mean_absolute_error(y_test, search_svr_linear.predict(x_test)),
           mean_absolute_error(y_test, search_svr_poly.predict(x_test)),
           mean_absolute_error(y_test, search_svr_rbf.predict(x_test))],
          [mean_absolute_percentage_error(y_test, search_linearsvr.predict(x_test)),
           mean_absolute_percentage_error(y_test, search_svr_linear.predict(x_test)),
           mean_absolute_percentage_error(y_test, search_svr_poly.predict(x_test)),
           mean_absolute_percentage_error(y_test, search_svr_rbf.predict(x_test))]],
    index=['R2', 'RMSE', 'MAE', "MAPE"],
    columns=["LinearSVR", "SVR_Linear", "SVR_poly", "SVR_rbf"])
print(pd_metrics_search1)
print('//' * 100)
pd_metrics_search1.to_csv('out1.csv')
print(time.time() - g)
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 10))

ax1.scatter(y_test, search_linearsvr.predict(x_test))
ax1.set_title('LinearSVR')
ax2.scatter(y_test, search_svr_linear.predict(x_test))
ax2.set_title('SVR linear')
ax3.scatter(y_test, search_svr_poly.predict(x_test))
ax3.set_title('SVR poly')
ax4.scatter(y_test, search_svr_rbf.predict(x_test))
ax4.set_title('SVR rbf')

# plt.show()
plt.savefig('saved_figure1.png')

pipe_linearsvr2 = Pipeline([('scaler', StandardScaler()), ('linear_svr', LinearSVR())])
pipe_svr_linear2 = Pipeline([('scaler', StandardScaler()), ('linear_svr', SVR(kernel='linear'))])
pipe_svr_poly2 = Pipeline([('scaler', StandardScaler()), ('linear_svr', SVR(kernel='poly'))])
pipe_svr_rbf2 = Pipeline([('scaler', StandardScaler()), ('linear_svr', SVR(kernel='rbf'))])

search_linearsvr2 = GridSearchCV(pipe_linearsvr2, {'linear_svr__epsilon': np.arange(0.5, 0.75, 0.005)})
search_svr_linear2 = GridSearchCV(pipe_svr_linear2, {'linear_svr__epsilon': np.arange(0.5, 0.75, 0.005)})
search_svr_poly2 = GridSearchCV(pipe_svr_poly2, {'linear_svr__epsilon': np.arange(0.5, 0.75, 0.001),
                                                 'linear_svr__C': np.arange(0.5, 10, 0.5),
                                                 'linear_svr__degree': [2, 3, 4, 5]})
search_svr_rbf2 = GridSearchCV(pipe_svr_rbf2, {'linear_svr__gamma': np.arange(0.5, 0.75, 0.005),
                                               'linear_svr__C': np.arange(0.5, 10, 0.5)})

search_linearsvr2.fit(x_train, y_train)
search_svr_linear2.fit(x_train, y_train)
search_svr_poly2.fit(x_train, y_train)
search_svr_rbf2.fit(x_train, y_train)

print(search_linearsvr2.best_params_)
print(search_svr_linear2.best_params_)
print(search_svr_poly2.best_params_)
print(search_svr_rbf2.best_params_)

pd_metrics_search2 = pd.DataFrame(
    data=[[search_linearsvr.score(x_test, y_test), search_svr_linear.score(x_test, y_test),
           search_svr_poly.score(x_test, y_test), search_svr_rbf.score(x_test, y_test)],
          [mean_squared_error(y_test, search_linearsvr.predict(x_test)),
           mean_squared_error(y_test, search_svr_linear.predict(x_test)),
           mean_squared_error(y_test, search_svr_poly.predict(x_test)),
           mean_squared_error(y_test, search_svr_rbf.predict(x_test))],
          [mean_absolute_error(y_test, search_linearsvr.predict(x_test)),
           mean_absolute_error(y_test, search_svr_linear.predict(x_test)),
           mean_absolute_error(y_test, search_svr_poly.predict(x_test)),
           mean_absolute_error(y_test, search_svr_rbf.predict(x_test))],
          [mean_absolute_percentage_error(y_test, search_linearsvr.predict(x_test)),
           mean_absolute_percentage_error(y_test, search_svr_linear.predict(x_test)),
           mean_absolute_percentage_error(y_test, search_svr_poly.predict(x_test)),
           mean_absolute_percentage_error(y_test, search_svr_rbf.predict(x_test))]],
    index=['R2', 'RMSE', 'MAE', "MAPE"],
    columns=["LinearSVR", "SVR_Linear", "SVR_poly", "SVR_rbf"])
print(pd_metrics_search1)
print('//' * 100)
pd_metrics_search2.to_csv('out2.csv')
print(time.time() - g)
fig2, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 10))

ax1.scatter(y_test, search_linearsvr.predict(x_test))
ax1.set_title('LinearSVR')
ax2.scatter(y_test, search_svr_linear.predict(x_test))
ax2.set_title('SVR linear')
ax3.scatter(y_test, search_svr_poly.predict(x_test))
ax3.set_title('SVR poly')
ax4.scatter(y_test, search_svr_rbf.predict(x_test))
ax4.set_title('SVR rbf')

# plt.show()
plt.savefig('saved_figure2.png')
