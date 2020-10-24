import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV

from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score

# from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
import pydot
from IPython.display import Image

# import keras

import warnings
import time

warnings.filterwarnings('ignore')


def null_detect(df, axis_n):
    """
    空值查找

    1为行
    0为列

    """
    df = df.isnull()
    df = df.sum(axis=axis_n)
    df = df.to_frame()
    df = df.sort_values(by=0, ascending=False)
    return df


def fillna_mean(df, col_name):
    """
    空值填充为均值
    """
    df[col_name] = df[col_name].fillna(df[col_name].mean())
    return df


def fillna_medians(df, col_name):
    """
    空值填充为中位数
    """
    df[col_name] = df[col_name].fillna(df[col_name].median())
    return df


def fillna_mode(df, col_name):
    """
    空值填充为众数
    """
    import scipy.stats as st
    df[col_name] = df[col_name].fillna(st.mode(df[col_name])[0][0])
    return df


data = pd.read_csv('5650final/CHD.csv')
# data_ind = data.drop(['education'], axis=1)
data_ind_nn = data.dropna(axis='rows')

sns.set_style('whitegrid')
sns.countplot(x='TenYearCHD', data=data_ind_nn)

"""
re-balance
"""
independent_vars = data_ind_nn.drop("TenYearCHD", 1)  # independent variables
dependent_var = data_ind_nn["TenYearCHD"]  # dependent variable
x, x_holdout, y, y_holdout = train_test_split(independent_vars, dependent_var, test_size=0.2, random_state=100)
data_ind_br = pd.concat([x, y], axis=1)
data_ind_br['TenYearCHD'].value_counts()

ros = RandomOverSampler()
x_ros, y_ros = ros.fit_sample(x, y)
data_ros = pd.concat([pd.DataFrame(x_ros), pd.DataFrame(y_ros)], axis=1)
data_ros.columns = x.columns.tolist() + ['Class']
data_ros = data_ros.reset_index()
print("before ROS the distribution is :")
print(y.value_counts())
print("\nafter ROS the distribution is :")
print(data_ros.iloc[:, -1].value_counts())

"""
feature selection
"""
x_rs = data_ros.drop("Class", 1)  # independent variables
y_rs = data_ros["Class"]  # dependent variable

# no of features
nof_list = np.arange(1, 10)
high_score = 0
# Variable to store the optimum features
nof = 0
score_list = []
x_train_rs, x_test_rs, y_train_rs, y_test_rs = train_test_split(x_rs, y_rs, test_size=0.2, random_state=0)
for n in range(len(nof_list)):
    model = DecisionTreeClassifier(random_state=10)
    # model= DecisionTreeClassifier
    # rfe: recursive feature elimination
    rfe = RFE(model, nof_list[n])
    x_train_rfe_rs = rfe.fit_transform(x_train_rs, y_train_rs)
    x_test_rfe_rs = rfe.transform(x_test_rs)
    model.fit(x_train_rfe_rs, y_train_rs)
    score = model.score(x_test_rfe_rs, y_test_rs)
    score_list.append(score)
    if (score > high_score):
        high_score = score
        nof = nof_list[n]
print("Optimum number of features: %d" % nof)
print("Score with %d features: %f" % (nof, high_score))

cols = list(x.columns)
model = DecisionTreeClassifier()
"""
Initializing the RFE model with optimum number of features
"""
rfe = RFE(model, nof)
"""
Transformation
"""
x_rfe = rfe.fit_transform(x, y)
"""
fitting by the model
"""
model.fit(x_rfe, y)
temp = pd.Series(rfe.support_, index=cols)
selected_features_rfe = temp[temp == True].index
selected_features_rfe = pd.DataFrame(selected_features_rfe, columns=['features'])


"""
建模部分
"""
data_ros_run = data_ros.drop(['male', 'currentSmoker', 'cigsPerDay', 'BPMeds', 'prevalentStroke', 'prevalentHyp',
                              'diabetes'], axis=1)
x = data_ros_run.drop("Class", 1)
y = data_ros_run["Class"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)
data = pd.concat([x_train, y_train], axis=1)


def model_fitting(model, modelname):
    start = time.time()
    model_fit = model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print('=====================================================================')
    print(modelname)
    confusion_mtx = confusion_matrix(y_true=y_test, y_pred=y_pred, labels=[0, 1])
    # print('Confusion matrix:\n', confusion_mtx)
    print(metrics.classification_report(y_test, y_pred))
    roc_auc = roc_auc_score(y_test, y_pred)
    print('Roc accuracy:')
    print(roc_auc)
    end = time.time()
    print('Running time:')
    print(str(end - start))
    return model_fit
    print('=====================================================================')


# model = [
#     [KNeighborsClassifier(n_neighbors=3), 'KNeighborsClassifier'], #未进行归一化
#     [LogisticRegression(solver='lbfgs'), 'LogisticRegression'],
#     [DecisionTreeClassifier(), 'DecisionTreeClassifier'],
#     [RandomForestClassifier(),'RandomForestClassifier'],
# ]


"""
KNN部分测试
"""
# for i in range(2,8):
#     model_fitting(KNeighborsClassifier(n_neighbors=i, weights='distance'), 'KNeighborsClassifier'+str(i))
#     """
#     效果不如uniform
#     """

for i in range(2,8):
    model_fitting(KNeighborsClassifier(n_neighbors=i), 'KNeighborsClassifier'+str(i))

"""
logistic regression部分测试
"""

solver = ['lbfgs', 'newton-cg', 'sag', 'saga']
for i in solver:
    model_fitting(LogisticRegression(solver=i), 'LogisticRegression with ' + i)
for i in solver:
    model_fitting(LogisticRegressionCV(solver=i, cv=5), 'LogisticRegressionCV with ' + i)


"""
decision tree部分
"""
model_fit = model_fitting(DecisionTreeClassifier(), 'DecisionTreeClassifier')
model_fit
# model_fit.get_depth()
# model_fit.get_n_leaves()



model_fitting(RandomForestClassifier(), 'RandomForestClassifier')


# for a in model:
#     model = a[0]
#     model.fit(x_train, y_train)  # step 2: fit
#     y_pred = model.predict(x_test)  # step 3: predict
#     print(f'{a[1]}')  #
#     confusion_mtx = confusion_matrix(y_true=y_test, y_pred=y_pred, labels=[0, 1])
#     print('Confusion matrix:\n', confusion_mtx)
#     labels = ['bad', 'good']
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     cax = ax.matshow(confusion_mtx, cmap=plt.cm.Blues)
#     fig.colorbar(cax)
#     ax.set_xticklabels([''] + labels)
#     ax.set_yticklabels([''] + labels)
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#     plt.show()
#     print(metrics.classification_report(y_test, y_pred))
#     cv_scores = cross_val_score(model, x_train, y_train, cv=5)  # cross-validation scores
#     print("Cross Validation Scores:", cv_scores)
#     roc_auc = roc_auc_score(y_test, y_pred)
#     fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(x_test)[:, 1])
#     plt.figure()
#     plt.plot(fpr, tpr, label=f'{a[1]} ' + '(area = %0.2f)' % roc_auc)
#     plt.plot([0, 1], [0, 1], 'r--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver operating characteristic')
#     plt.legend(loc="lower right")
#     plt.savefig('Log_ROC')
#     # plt.show()
#
#     # print('-' * 100)