# %%
# Импортируем нужные библиотеки
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn import metrics, model_selection
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import statsmodels.api as sm
import scipy as sci
import time
start_time = time.time()


# %%
# Читаем данные и смотрим названия столбцов и их первые значения
data = pd.read_csv('data\Physical_parameters_.csv')
data.head()


# %%
# Дропаем значения, которые указывают погрешность, поскольку не будем ее учитывать и названия звезд, они нам не пригодятя
data = data.drop(['Star','e_Teff', 'e_logg','e_Vt','e_[Fe/H]','e_Mass','e_Age'], axis=1)
data.head()


# %%
#Заменяем пустые значениями NaN и смотрим типы значений столбцов
data[data == "     "] = np.nan
data[data == "      "] = np.nan
data.dtypes


# %%
#Уточняем типы данных в столбцах
data["Mass"] = data["Mass"].astype(float)
data["Age"] = data["Age"].astype(float)
data.dtypes


# %%
#Узнаем количество нулевых значений
data.isnull().sum()


# %%
#Удаляем строки, содержащие нулевые значения
data = data.dropna(axis=0, how='any', inplace=False)
data.head()


# %%
#Узнаем основные параметры
data.describe()


# %%
#Визуализируем датасет 
sns.pairplot(data)


# %%
#Нормализуем датасет
scaler = MinMaxScaler()


data[['Teff','logg','Vt','[Fe/H]','Mass','Age']] = scaler.fit_transform(data[['Teff','logg','Vt','[Fe/H]','Mass','Age']])


data.head()


# %%
#Строим матрицу кореляции


sns.heatmap(data.corr(), vmin=-1., vmax=1., annot=True, fmt='.2f', cmap="YlGnBu", cbar=True, linewidths=0.5, square=True)


# %%
#Оценим важность фич


X, y = data.drop('Age', axis=1), data['Age']


rfc = RandomForestRegressor()


rfc.fit(X, y)


plt.figure(figsize=(10, 6))


feat_importances = pd.Series(rfc.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')


plt.xlabel('Importance')
plt.title('Feature importance')


plt.show()


# %%
#Ненужных фич нет, поэтому дальше начинаем решать задачу регрессии, деля данные на трейн и тест


X_train, X_test = model_selection.train_test_split(data, test_size=0.3)


y_train = X_train['Age']
y_test = X_test['Age']


X_train.drop(columns=['Age'], inplace=True)
X_test.drop(columns=['Age'], inplace=True)
X_train.head()


# %%
fig = plt.figure(figsize=(20, 20))
axes = sns.jointplot(
    x=X_train['Teff'], y=y_train,
    kind='reg',
    ci=95)
plt.show()


# %%
fig = plt.figure(figsize=(20, 20))
axes = sns.jointplot(
    x=X_train['logg'], y=y_train,
    kind='reg',
    ci=95)
plt.show()


# %%
fig = plt.figure(figsize=(20, 20))
axes = sns.jointplot(
    x=X_train['Vt'], y=y_train,
    kind='reg',
    ci=95)
plt.show()


# %%
fig = plt.figure(figsize=(20, 20))
axes = sns.jointplot(
    x=X_train['[Fe/H]'], y=y_train,
    kind='reg',
    ci=95)
plt.show()


# %%
fig = plt.figure(figsize=(20, 20))
axes = sns.jointplot(
    x=X_train['Mass'], y=y_train,
    kind='reg',
    ci=95)
plt.show()


# %%
#Создаем и обучаем линейную регрессию
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# %%
#Смотрим весовые коэффициенты линейной регрессии, которые мы в результате получили
coeff_df = pd.DataFrame(regressor.coef_, X_train.columns, columns=['Coefficient'])
coeff_df


# %%
#Делаем предсказания на тестовом датафрейме
y_pred = regressor.predict(X_test)


# %%
#Выводим реальное значение целевой переменной (Actual) и предсказанное (Predicted)


p_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
p_df


# %%
#Смотрим метрики - ошибку между реальным и предсказанным значением


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


relative_error = np.sqrt(metrics.mean_squared_error(y_test, y_pred)) / np.mean(y_test) * 100
print(relative_error,'%')


# %%
#Пробуем обучить линейную гребневую регрессию


regressor1 = Ridge(alpha=0.5)
regressor1.fit(X_train, y_train)


# %%
#Реальное и предсказанное
y_pred1 = regressor1.predict(X_test)
p_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred1})
p_df


# %%
#Метрики


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred1))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred1))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred1)))


relative_error = np.sqrt(metrics.mean_squared_error(y_test, y_pred1)) / np.mean(y_test) * 100
print(relative_error,'%')


# %%
#Метод k ближайщих соседей


knn = KNeighborsRegressor(n_neighbors=6)
knn.fit(X_train, y_train)


# %%
#Реальное и предсказанное
y_pred2 = knn.predict(X_test)
p_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred2})
p_df


# %%
#Метрики
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred2))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred2))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred2)))


relative_error = np.sqrt(metrics.mean_squared_error(y_test, y_pred2)) / np.mean(y_test) * 100
print(relative_error,'%')


# %%
#Метод решающих деревьев  


tree = DecisionTreeRegressor(max_depth=3)
tree.fit(X_train, y_train)


# %%
#Реальное и предсказанное
y_pred3 = knn.predict(X_test)
p_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred3})
p_df


# %%
#Метрики
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred3))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred3))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred3)))


relative_error = np.sqrt(metrics.mean_squared_error(y_test, y_pred3)) / np.mean(y_test) * 100
print(relative_error,'%')


# %%
# Еще один способ обучить линейную регрессию - лежит в statsmodels, но он более конкретный и для него нужно строить линейную регрессию для каждого параметры отдельно


# Доверительная вероятность и уровень значимости:
p_level = 0.95
a_level = 1 - p_level
#R2 метрика, коэффициент  детерминации, показывающий насколько расчетные параметры модели объясняют зависимость и изменения целевой переменной Y от нецелевых фич - X.


#Можно сказать что, это показатель качества модели и чем он выше тем лучше.
#Понятное дело, что он не может быть больше 1 и считается неплохо, когда R2 выше 0,8, а если меньше 0,5, то смысл такой модели можно смело ставить под большой вопрос.


# Используем scipy для подробных метрик
def regression_model_adequacy_check(
        model_fit,
        p_level: float=0.95,
        model_name=''):


    n = int(model_fit.nobs)
    p = int(model_fit.df_model)  


    SST = model_fit.centered_tss 
    dfT = n-1
    MST = SST / dfT


    SSE = model_fit.ssr 
    dfE = n - p - 1
    MSE = SSE / dfE


    F_calc = MST / MSE
    F_table = sci.stats.f.ppf(p_level, dfT, dfE, loc=0, scale=1)
    a_calc = 1 - sci.stats.f.cdf(F_calc, dfT, dfE, loc=0, scale=1)
    conclusion_model_adequacy_check = 'adequacy' if F_calc >= F_table else 'non adequacy'


    result = pd.DataFrame({
        'SST': (SST),
        'SSE': (SSE),
        'dfT': (dfT),
        'dfE': (dfE),
        'MST': (MST),
        'MSE': (MSE),
        'p_level': (p_level),
        'a_level': (a_level),
        'F_calc': (F_calc),
        'F_table': (F_table),
        'F_calc >= F_table': (F_calc >= F_table),
        'a_calc': (a_calc),
        'a_calc <= a_level': (a_calc <= a_level),
        'adequacy_check': (conclusion_model_adequacy_check),
    },
        index=[model_name]
    )


    return result


# %%


def  simple_linear_regression(X_train, y_train, X_test, y_test, column_name):
 
   X = X_train[[column_name]]
   y = y_train


   model = sm.OLS(y, X)
   results = model.fit()


   results.summary()


   fig, ax = plt.subplots()
   fig = sm.graphics.plot_fit(results, 0, ax=ax)
   ax.set_ylabel("body_mass_g")
   ax.set_xlabel(column_name)
   ax.set_title("Linear Regression")


   print('R2 =', results.rsquared)


   res = regression_model_adequacy_check(results, p_level=0.95, model_name='linear_ols')
   print(res)
   
simple_linear_regression(X_train, y_train, X_test, y_test, 'Teff' )


# %%
simple_linear_regression(X_train, y_train, X_test, y_test, 'logg' )


# %%
simple_linear_regression(X_train, y_train, X_test, y_test, 'Vt' )


# %%
simple_linear_regression(X_train, y_train, X_test, y_test, '[Fe/H]' )


# %%
simple_linear_regression(X_train, y_train, X_test, y_test, 'Mass' )


# %%
print("--- %s seconds ---" % (time.time() - start_time))
