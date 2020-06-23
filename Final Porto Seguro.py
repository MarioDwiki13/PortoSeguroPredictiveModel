import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

#Data Setup
x_train = pd.read_csv('train.csv')
x_test = pd.read_csv('test.csv')
y_train = x_train['target'].ravel()

del x_train['target']
del x_train['id']
del x_test['id']
del x_train['ps_car_03_cat']
del x_test['ps_car_03_cat']
del x_train['ps_car_05_cat']
del x_test['ps_car_05_cat']

x_train = x_train.replace(-1,np.NaN)

for i in list(x_train):
    mean = x_train[i].mean()
    median = x_train[i].quantile(q=0.5)
    if 'bin' in i:
        if x_train[i].quantile(q=0.5)==0:
            x_train[i].replace(np.NaN,0)
        else:
            x_train[i].replace(np.NaN,1)
    elif x_train[i].dtype=='int64':
        x_train[i] = x_train[i].replace(np.NaN,median)
    elif x_train[i].dtype=='float64':
        x_train[i] = x_train[i].replace(np.NaN,mean)

input_train, input_test, output_train, output_test = train_test_split(x_train, y_train, test_size = 0.33, random_state = 42)

#Logistic Regression
log = LogisticRegression(solver = 'newton-cg', max_iter = 100, multi_class = 'ovr', n_jobs = 1)
log_cv = GridSearchCV(log, param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}, cv = 3)
log_cv.fit(input_train, output_train)
log_result = log_cv.predict_proba(x_test)

temp_log = []
for i in range(len(x_test)):
    temp_log.append(log_result[i][1])

print 'Logistic Regression'
print 'Parameter terbaik: %s'%log_cv.best_params_
print 'Estimasi akurasi: %s'%log_cv.best_score_
print 'Confusion matriks:'
print confusion_matrix(output_test, log_cv.predict(input_test))
print '\n'

#Decision Tree
tree = DecisionTreeClassifier(criterion = 'gini', splitter = 'best')
tree_cv = GridSearchCV(tree, param_grid = {'max_depth': np.arange(3, 11)}, cv = 3)
tree_cv.fit(input_train, output_train)
tree_result = tree_cv.predict_proba(x_test)

temp_tree = []
for i in range(len(x_test)):
    temp_tree.append(tree_result[i][1])

print 'Decision Tree'
print 'Parameter terbaik: %s'%tree_cv.best_params_
print 'Estimasi akurasi: %s'%tree_cv.best_score_
print 'Confusion matriks:'
print confusion_matrix(output_test, tree_cv.predict(input_test))
print '\n'

#Random Forest
forest = RandomForestClassifier(n_estimators = 50, bootstrap = False, n_jobs = -1)
forest_cv = GridSearchCV(forest, param_grid = {'max_depth': np.arange(3, 11)}, cv = 3)
forest_cv.fit(input_train, output_train)
forest_result = forest_cv.predict_proba(x_test)

temp_forest = []
for i in range(len(x_test)):
    temp_forest.append(forest_result[i][1])

print 'Random Forest'
print 'Parameter terbaik: %s'%forest_cv.best_params_
print 'Estimasi akurasi: %s'%forest_cv.best_score_
print 'Confusion matriks:'
print confusion_matrix(output_test, forest_cv.predict(input_test))
print '\n'

#Neural Network
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

nn = MLPClassifier()
nn_cv = GridSearchCV(nn, param_grid = {'alpha': 10.0 ** -np.arange(1,7)}, cv = 3)
nn_cv.fit(x_train, y_train)
nn_result = nn_cv.predict_proba(x_test)

temp_nn = []
for i in range(len(x_test)):
    temp_nn.append(nn_result[i][1])

print 'Neural Network'
print 'Parameter terbaik: %s'%nn_cv.best_params_
print 'Estimasi akurasi: %s'%nn_cv.best_score_
print 'Confusion matriks:'
print confusion_matrix(output_test, nn_cv.predict(input_test))
print '\n'

#Result
result = []
for i in range(len(x_test)):
    result.append([i,round((temp_log[i]+temp_tree[i]+temp_forest[i]+temp_nn[i])/4,4)])

#Export to CSV
with open('Output.csv','w') as output:
    writer = csv.writer(output, lineterminator = '\n')
    writer.writerows(result)
