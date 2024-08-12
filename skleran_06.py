import numpy as np
from graphviz import Digraph
from scipy import sparse
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
import mglearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import make_blobs
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

# Windows의 경우
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc("font", family=font_name)

# 음수 기호가 깨지는 문제 해결
rc("axes", unicode_minus=False)

from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X, y = make_blobs(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

logreg = LogisticRegression().fit(X_train, y_train)

print("테스트 세트 점수: {:.2f}".format(logreg.score(X_test, y_test)))

mglearn.plots.plot_cross_validation()
plt.show()

from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

iris = load_iris()
logreg = LogisticRegression(max_iter=1000)

scores = cross_val_score(logreg, iris.data, iris.target)
print("교차 검증 점수: {}".format(scores))

scores = cross_val_score(logreg, iris.data, iris.target, cv=5)
print("교차 검증 점수: {}".format(scores))

print("교차 검증 평균 점수: {:.2f}".format(scores.mean()))

from sklearn.model_selection import cross_validate

res = cross_validate(logreg, iris.data, iris.target, cv=5, return_train_score=True)
display(pd.DataFrame(res))
res = cross_validate(logreg, iris.data, iris.target, cv=5, return_train_score=True)
display(pd.DataFrame(res))

from sklearn.model_selection import KFold

kfold = KFold(n_splits=5)
print(
    "교차 검증 점수:\n{}".format(
        cross_val_score(logreg, iris.data, iris.target, cv=kfold)
    )
)

kfold = KFold(n_splits=3)
print(
    "교차 검증 점수:\n{}".format(
        cross_val_score(logreg, iris.data, iris.target, cv=kfold)
    )
)

kfold = KFold(n_splits=3, shuffle=True, random_state=0)
print(
    "교차 검증 점수:\n{}".format(
        cross_val_score(logreg, iris.data, iris.target, cv=kfold)
    )
)

from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
scores = cross_val_score(logreg, iris.data, iris.target, cv=loo)
print("교차 검증 분할 횟수: ", len(scores))
print("평균 정확도: {:.2f}".format(scores.mean()))

mglearn.plots.plot_shuffle_split()
plt.show()

from sklearn.model_selection import ShuffleSplit

shuffle_split = ShuffleSplit(test_size=0.5, train_size=0.5, n_splits=10)
scores = cross_val_score(logreg, iris.data, iris.target, cv=shuffle_split)
print("교차 검증 점수:\n{}".format(scores))

from sklearn.model_selection import GroupKFold

X, y = make_blobs(n_samples=12, random_state=0)
groups = [0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3]
scores = cross_val_score(logreg, X, y, groups, cv=GroupKFold(n_splits=3))
print("교차 검증 점수:\n{}".format(scores))

mglearn.plots.plot_group_kfold()
plt.show()

mglearn.plots.plot_stratified_cross_validation()
plt.show()

from sklearn.model_selection import KFold

kfold = KFold(n_splits=5)
print(
    "교차 검증 점수:\n{}".format(
        cross_val_score(logreg, iris.data, iris.target, cv=kfold)
    )
)

kfold = KFold(n_splits=3)
print(
    "교차 검증 점수:\n{}".format(
        cross_val_score(logreg, iris.data, iris.target, cv=kfold)
    )
)

from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()

scores = cross_val_score(logreg, iris.data, iris.target, cv=loo)
print("교차 검증 분할 횟수: ", len(scores))
print("평균 정확도: {:.2f}".format(scores.mean()))

mglearn.plots.plot_shuffle_split()
plt.show()

from sklearn.model_selection import ShuffleSplit

shuffle_split = ShuffleSplit(test_size=0.5, train_size=0.5, n_splits=10)
scores = cross_val_score(logreg, iris.data, iris.target, cv=shuffle_split)
print("교차 검증 점수:\n{}".format(scores))

from sklearn.svm import SVC

X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, random_state=0
)
best_score = 0

for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:

    for C in [0.001, 0.01, 0.1, 1, 10, 100]:

        svm = SVC(gamma=gamma, C=C)
        svm.fit(X_train, y_train)
        score = svm.score(X_test, y_test)

        if score > best_score:
            best_score = score
            best_parameters = {"C": C, "gamma": gamma}

print("최고 점수: {:.2f}".format(best_score))
print("최적 매개변수: {}".format(best_parameters))

mglearn.plots.plot_threefold_split()
plt.show()

X_trainval, X_test, y_trainval, y_test = train_test_split(
    iris.data, iris.target, random_state=0
)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_trainval, y_trainval, random_state=1
)
print(
    "훈련 세트의 크기: {}\n검증 세트의 크기: {}\n테스트 세트의 크기: {}\n".format(
        X_train.shape[0], X_valid.shape[0], X_test.shape[0]
    )
)

best_score = 0

for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:

    for C in [0.001, 0.01, 0.1, 1, 10, 100]:

        svm = SVC(gamma=gamma, C=C)
        svm.fit(X_train, y_train)
        score = svm.score(X_valid, y_valid)

        if score > best_score:
            best_score = score
            best_parameters = {"C": C, "gamma": gamma}

svm = SVC(**best_parameters)
svm.fit(X_trainval, y_trainval)
test_score = svm.score(X_test, y_test)

print("검증 세트에서 최고 점수: {:.2f}".format(best_score))
print("최적 매개변수: ", best_parameters)
print("최적 매개변수에서 테스트 세트 점수: {:.2f}".format(test_score))
