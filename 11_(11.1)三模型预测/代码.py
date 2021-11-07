import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

data_1 = pd.read_excel("data1.xlsx")
data_2 = pd.read_excel("data2.xlsx")


def ModelPredict(data):
    # 数据分割 X为训练集
    X = data.iloc[:6, 1:3]
    y_a = data.iloc[:6, 3:4]
    y_b = data.iloc[:6, 4:5]
    y_c = data.iloc[:6, 5:6]
    y_B = data.iloc[:6, 6:7]
    y_G = data.iloc[:6, 7:8]
    test_x = data.iloc[6:7, 1:3]

    # 多元线性回归预测
    model = LinearRegression()
    model.fit(X, y_a)
    pre_a = model.predict(test_x)
    print("回归模型有关a的预测中线性参数为", model.coef_, "截距为", model.intercept_)
    model.fit(X, y_b)
    pre_b = model.predict(test_x)
    print("回归模型有关b的预测中线性参数为", model.coef_, "截距为", model.intercept_)
    model.fit(X, y_c)
    pre_c = model.predict(test_x)
    print("回归模型有关c的预测中线性参数为", model.coef_, "截距为", model.intercept_)

    # GBDT预测
    model = GradientBoostingRegressor()
    # Choose all predictors except target & IDcols
    param_test1 = {'n_estimators': range(1, 100)}
    params = range(1, 100)
    gsearch1 = GridSearchCV(estimator=model, param_grid=param_test1, scoring='neg_mean_absolute_error')
    gsearch1.fit(X, y_B)  # #XGBOOST预测
    print(gsearch1.cv_results_['params'], gsearch1.cv_results_['mean_test_score'])
    plt.figure()
    plt.plot(params, gsearch1.cv_results_['mean_test_score'])
    plt.title("GBDT")
    plt.show()

    #     XGBOOST预测
    model = xgb.XGBRegressor()
    param_test1 = {'n_estimators': range(1, 100)}
    params = range(1, 100)
    gsearch1 = GridSearchCV(estimator=model, param_grid=param_test1, scoring='neg_mean_absolute_error')
    gsearch1.fit(X, y_B)  # #XGBOOST预测
    print(gsearch1.cv_results_['params'], gsearch1.cv_results_['mean_test_score'])
    plt.figure()
    plt.plot(params, gsearch1.cv_results_['mean_test_score'])
    plt.title("XGBOOST")
    plt.show()

ModelPredict(data_1)
ModelPredict(data_2)
