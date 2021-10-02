Python 3.9.5 (tags/v3.9.5:0a7dcbd, May  3 2021, 17:27:52) [MSC v.1928 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> 
= RESTART: C:/Users/Adithya Ramgiri/AppData/Local/Programs/Python/Python39/project.py
exit()
>>> import numpy as np
>>> import pandas as pd
>>> import sklearn
>>> from sklearn.datasets import load_boston
>>> df=load_boston()
>>> print(df.feature_names)
['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
 'B' 'LSTAT']
>>> df.keys()
dict_keys(['data', 'target', 'feature_names', 'DESCR', 'filename'])
>>> boston = pd.DataFrame(df.data, columns=df.feature_names)
>>> boston.head()
      CRIM    ZN  INDUS  CHAS    NOX  ...  RAD    TAX  PTRATIO       B  LSTAT
0  0.00632  18.0   2.31   0.0  0.538  ...  1.0  296.0     15.3  396.90   4.98
1  0.02731   0.0   7.07   0.0  0.469  ...  2.0  242.0     17.8  396.90   9.14
2  0.02729   0.0   7.07   0.0  0.469  ...  2.0  242.0     17.8  392.83   4.03
3  0.03237   0.0   2.18   0.0  0.458  ...  3.0  222.0     18.7  394.63   2.94
4  0.06905   0.0   2.18   0.0  0.458  ...  3.0  222.0     18.7  396.90   5.33

[5 rows x 13 columns]
>>> boston.head(16)
       CRIM    ZN  INDUS  CHAS    NOX  ...  RAD    TAX  PTRATIO       B  LSTAT
0   0.00632  18.0   2.31   0.0  0.538  ...  1.0  296.0     15.3  396.90   4.98
1   0.02731   0.0   7.07   0.0  0.469  ...  2.0  242.0     17.8  396.90   9.14
2   0.02729   0.0   7.07   0.0  0.469  ...  2.0  242.0     17.8  392.83   4.03
3   0.03237   0.0   2.18   0.0  0.458  ...  3.0  222.0     18.7  394.63   2.94
4   0.06905   0.0   2.18   0.0  0.458  ...  3.0  222.0     18.7  396.90   5.33
5   0.02985   0.0   2.18   0.0  0.458  ...  3.0  222.0     18.7  394.12   5.21
6   0.08829  12.5   7.87   0.0  0.524  ...  5.0  311.0     15.2  395.60  12.43
7   0.14455  12.5   7.87   0.0  0.524  ...  5.0  311.0     15.2  396.90  19.15
8   0.21124  12.5   7.87   0.0  0.524  ...  5.0  311.0     15.2  386.63  29.93
9   0.17004  12.5   7.87   0.0  0.524  ...  5.0  311.0     15.2  386.71  17.10
10  0.22489  12.5   7.87   0.0  0.524  ...  5.0  311.0     15.2  392.52  20.45
11  0.11747  12.5   7.87   0.0  0.524  ...  5.0  311.0     15.2  396.90  13.27
12  0.09378  12.5   7.87   0.0  0.524  ...  5.0  311.0     15.2  390.50  15.71
13  0.62976   0.0   8.14   0.0  0.538  ...  4.0  307.0     21.0  396.90   8.26
14  0.63796   0.0   8.14   0.0  0.538  ...  4.0  307.0     21.0  380.02  10.26
15  0.62739   0.0   8.14   0.0  0.538  ...  4.0  307.0     21.0  395.62   8.47

[16 rows x 13 columns]
>>> boston['MEDV'] = df.target
>>> boston.head()
      CRIM    ZN  INDUS  CHAS    NOX  ...    TAX  PTRATIO       B  LSTAT  MEDV
0  0.00632  18.0   2.31   0.0  0.538  ...  296.0     15.3  396.90   4.98  24.0
1  0.02731   0.0   7.07   0.0  0.469  ...  242.0     17.8  396.90   9.14  21.6
2  0.02729   0.0   7.07   0.0  0.469  ...  242.0     17.8  392.83   4.03  34.7
3  0.03237   0.0   2.18   0.0  0.458  ...  222.0     18.7  394.63   2.94  33.4
4  0.06905   0.0   2.18   0.0  0.458  ...  222.0     18.7  396.90   5.33  36.2

[5 rows x 14 columns]
>>> boston.isnull()
      CRIM     ZN  INDUS   CHAS    NOX  ...    TAX  PTRATIO      B  LSTAT   MEDV
0    False  False  False  False  False  ...  False    False  False  False  False
1    False  False  False  False  False  ...  False    False  False  False  False
2    False  False  False  False  False  ...  False    False  False  False  False
3    False  False  False  False  False  ...  False    False  False  False  False
4    False  False  False  False  False  ...  False    False  False  False  False
..     ...    ...    ...    ...    ...  ...    ...      ...    ...    ...    ...
501  False  False  False  False  False  ...  False    False  False  False  False
502  False  False  False  False  False  ...  False    False  False  False  False
503  False  False  False  False  False  ...  False    False  False  False  False
504  False  False  False  False  False  ...  False    False  False  False  False
505  False  False  False  False  False  ...  False    False  False  False  False

[506 rows x 14 columns]
>>> boston.isnull().sum()
CRIM       0
ZN         0
INDUS      0
CHAS       0
NOX        0
RM         0
AGE        0
DIS        0
RAD        0
TAX        0
PTRATIO    0
B          0
LSTAT      0
MEDV       0
dtype: int64
>>> from sklearn.model_selection import train_test_split
>>> X=boston.drop('MEDV', axis =1)
>>> Y = boston['MEDV']
>>> X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=5)
>>>  print(X_train.shape)
 
SyntaxError: unexpected indent
>>> print(X_train.shape)
(430, 13)
>>> print(X_test.shape)
(76, 13)
>>> print(Y_train.shape)
(430,)
>>> print(Y_test.shape)
(76,)
>>> from sklearn.linear_model import LinearRegression
>>> from sklearn.metrics import mean_squared_error
>>> lin_model=LinearRegression()
>>> lin_model.fit(X_train, Y_train)
LinearRegression()
>>> y_train_predict = lin_model.predict(X_train)
>>> rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
>>> print("The model performce for training set")
The model performce for training set
>>> print('RMSE is{}'.format(rmse))
RMSE is4.710901797319796
>>> y_test_predict = lin_model.predict(X_test)
>>> rmse=(np.sqrt(mean_squared_error(Y_test, y_test_predict)))
>>> print("The model performce for testing set")
The model performce for testing set
>>> print('RMSE is{}'.format(rmse))
RMSE is4.687543527902935
>>>  import matplotlib.pyplot as plt
 
SyntaxError: unexpected indent
>>> import matplotlib.pyplot as plt
Traceback (most recent call last):
  File "<pyshell#39>", line 1, in <module>
    import matplotlib.pyplot as plt
ModuleNotFoundError: No module named 'matplotlib'
>>> import matplotlib.pyplot as plt
>>> plt.figure(figsize=(5,5))
<Figure size 500x500 with 0 Axes>
>>> plt.scartter(Y_test, y_test_predict)
Traceback (most recent call last):
  File "<pyshell#42>", line 1, in <module>
    plt.scartter(Y_test, y_test_predict)
AttributeError: module 'matplotlib.pyplot' has no attribute 'scartter'
>>> plt.scatter(Y_test, y_test_predict)
<matplotlib.collections.PathCollection object at 0x0000029BED00F2E0>
>>> plt.plot([min(y_test_predict),max(y_test_predict)],[min(y_test_predict),max(y_test_predict)])
[<matplotlib.lines.Line2D object at 0x0000029BED01B6D0>]
>>> plt.xlabel('Actual')
Text(0.5, 0, 'Actual')
>>> plt.ylabel('Predicted")
	   
SyntaxError: EOL while scanning string literal
>>> plt.ylabel('Predicted')
Text(0, 0.5, 'Predicted')
>>> 