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
from sklearn.metrics import precision_recall_curve
from mglearn.datasets import make_blobs
from sklearn.svm import SVC

# Windows의 경우
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc("font", family=font_name)

# 음수 기호가 깨지는 문제 해결
rc("axes", unicode_minus=False)


# 부드러운 곡선을 위해 데이터 포인트 수를 늘립니다
X, y = make_blobs(n_samples=(4000, 500), cluster_std=[7.0, 2], random_state=22)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
svc = SVC(gamma=0.05).fit(X_train, y_train)
precision, recall, thresholds = precision_recall_curve(
    y_test, svc.decision_function(X_test)
)
# 0에 가까운 임계값을 찾습니다
close_zero = np.argmin(np.abs(thresholds))

plt.plot(
    precision[close_zero],
    recall[close_zero],
    "o",
    markersize=10,
    label="임계값 0",
    fillstyle="none",
    c="k",
    mew=2,
)
plt.plot(precision, recall, label="정밀도-재현율 곡선")
plt.xlabel("정밀도")
plt.ylabel("재현율")
plt.legend(loc="best")
plt.show()

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=0, max_features=2)
rf.fit(X_train, y_train)

# RandomForestClassifier는 decision_function 대신 predict_proba를 제공합니다.
precision_rf, recall_rf, thresholds_rf = precision_recall_curve(
    y_test, rf.predict_proba(X_test)[:, 1]
)

plt.plot(precision, recall, label="svc")

plt.plot(
    precision[close_zero],
    recall[close_zero],
    "o",
    markersize=10,
    label="svc: 임계값 0",
    fillstyle="none",
    c="k",
    mew=2,
)

plt.plot(precision_rf, recall_rf, label="rf")

close_default_rf = np.argmin(np.abs(thresholds_rf - 0.5))
plt.plot(
    precision_rf[close_default_rf],
    recall_rf[close_default_rf],
    "^",
    c="k",
    markersize=10,
    label="rf: 임계값 0.5",
    fillstyle="none",
    mew=2,
)

plt.xlabel("정밀도")
plt.ylabel("재현율")
plt.legend(loc="best")

plt.show()

from sklearn.metrics import average_precision_score

ap_rf = average_precision_score(y_test, rf.predict_proba(X_test)[:, 1])
ap_svc = average_precision_score(y_test, svc.decision_function(X_test))
print("랜덤 포레스트의 평균 정밀도: {:.3f}".format(ap_rf))
print("SVC의 평균 정밀도: {:.3f}".format(ap_svc))

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, svc.decision_function(X_test))

plt.plot(fpr, tpr, label="ROC 곡선")
plt.xlabel("FPR")
plt.ylabel("TPR (재현율)")
# 0 근처의 임계값을 찾습니다
close_zero = np.argmin(np.abs(thresholds))
plt.plot(
    fpr[close_zero],
    tpr[close_zero],
    "o",
    markersize=10,
    label="임계값 0",
    fillstyle="none",
    c="k",
    mew=2,
)
plt.legend(loc=4)
plt.show()
