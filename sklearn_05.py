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

mglearn.datasets.DATA_PATH
import os

data = pd.read_csv(
    os.path.join(mglearn.datasets.DATA_PATH, "adult.data"),
    header=None,
    index_col=False,
    names=[
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "gender",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income",
    ],
)
data = data[
    [
        "age",
        "workclass",
        "education",
        "gender",
        "hours-per-week",
        "occupation",
        "income",
    ]
]
data.head()

print(data.gender.value_counts())
print("원본 특성:\n", list(data.columns), "\n")
data_dummies = pd.get_dummies(data)
print("get_dummies 후의 특성:\n", list(data_dummies.columns))

data_dummies.head()

features = data_dummies.loc[:, "age":"occupation_ Transport-moving"]
X = features.values
y = data_dummies["income_ >50K"].values
print("X.shape: {} y.shape: {}".format(X.shape, y.shape))

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
print("테스트 점수: {:.2f}".format(logreg.score(X_test, y_test)))

demo_df = pd.DataFrame(
    {"숫자 특성": [0, 1, 2, 1], "범주형 특성": ["양말", "여우", "양말", "상자"]}
)
demo_df
pd.get_dummies(demo_df)
demo_df["숫자 특성"] = demo_df["숫자 특성"].astype(str)
pd.get_dummies(demo_df)

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse_output=False)
print(ohe.fit_transform(demo_df))
print(ohe.get_feature_names_out())

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

ct = ColumnTransformer(
    [
        ("scaling", StandardScaler(), ["age", "hours-per-week"]),
        (
            "onehot",
            OneHotEncoder(sparse_output=False),
            ["workclass", "education", "gender", "occupation"],
        ),
    ]
)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data_features = data.drop("income", axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    data_features, data.income, random_state=0
)
ct.fit(X_train)

X_train_trans = ct.transform(X_train)
print(X_train_trans.shape)

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_trans, y_train)
X_test_trans = ct.transform(X_test)
print("테스트 점수: {:.2f}".format(logreg.score(X_test_trans, y_test)))

ct.named_transformers_.onehot

from sklearn.compose import make_column_transformer

ct = make_column_transformer(
    (StandardScaler(), ["age", "hours-per-week"]),
    (
        OneHotEncoder(sparse_output=False),
        ["workclass", "education", "gender", "occupation"],
    ),
)

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

X, y = mglearn.datasets.make_wave(n_samples=120)
line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)

reg = DecisionTreeRegressor(min_samples_split=3).fit(X, y)
plt.plot(line, reg.predict(line), label="결정 트리")
# plt.show()

reg = LinearRegression().fit(X, y)
plt.plot(line, reg.predict(line), label="선형 회귀")
# plt.show()

plt.plot(X[:, 0], y, "o", c="k")
plt.ylabel("회귀 출력")
plt.xlabel("입력 특성")
plt.legend(loc="best")
plt.show()

from sklearn.preprocessing import KBinsDiscretizer

kb = KBinsDiscretizer(n_bins=10, strategy="uniform")
kb.fit(X)
print("bin edges: \n", kb.bin_edges_)
X_binned = kb.transform(X)

print(X[:10])
print(X_binned.toarray()[:10])

kb = KBinsDiscretizer(n_bins=10, strategy="uniform", encode="onehot-dense")
kb.fit(X)
X_binned = kb.transform(X)

line_binned = kb.transform(line)

reg = LinearRegression().fit(X_binned, y)
plt.plot(line, reg.predict(line_binned), label="구간 선형 회귀")
reg = DecisionTreeRegressor(min_samples_split=3).fit(X_binned, y)
plt.plot(line, reg.predict(line_binned), label="구간 결정 트리")
plt.plot(X[:, 0], y, "o", c="k")
plt.vlines(kb.bin_edges_[0], -3, 3, linewidth=1, alpha=0.2)
plt.legend(loc="best")
plt.ylabel("회귀 출력")
plt.xlabel("입력 특성")
plt.show()

X_combined = np.hstack([X, X_binned])
print(X_combined.shape)

reg = LinearRegression().fit(X_combined, y)

line_combined = np.hstack([line, line_binned])
plt.plot(line, reg.predict(line_combined), label="원본 특성을 더한 선형 회귀")
plt.vlines(kb.bin_edges_[0], -3, 3, linewidth=1, alpha=0.2)
plt.legend(loc="best")
plt.ylabel("회귀 출력")
plt.xlabel("입력 특성")
plt.plot(X[:, 0], y, "o", c="k")
plt.show()

X_product = np.hstack([X_binned, X * X_binned])
print(X_product.shape)

reg = LinearRegression().fit(X_product, y)
line_product = np.hstack([line_binned, line * line_binned])
plt.plot(line, reg.predict(line_product), label="원본 특성을 곱한 선형 회귀")
plt.vlines(kb.bin_edges_[0], -3, 3, linewidth=1, alpha=0.2)
plt.plot(X[:, 0], y, "o", c="k")
plt.ylabel("회귀 출력")
plt.xlabel("입력 특성")
plt.legend(loc="best")
plt.show()

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=10, include_bias=False)
poly.fit(X)
X_poly = poly.transform(X)

print("X_poly.shape:", X_poly.shape)

print("X 원소:\n", X[:5])
print("X_poly 원소:\n", X_poly[:5])

print("다항 특성 이름:\n", poly.get_feature_names_out())

reg = LinearRegression().fit(X_poly, y)

line_poly = poly.transform(line)
plt.plot(line, reg.predict(line_poly), label="다항 선형 회귀")
plt.plot(X[:, 0], y, "o", c="k")
plt.ylabel("회귀 출력")
plt.xlabel("입력 특성")
plt.legend(loc="best")
plt.show()

from sklearn.svm import SVR

for gamma in [1, 10]:
    svr = SVR(gamma=gamma).fit(X, y)
    plt.plot(line, svr.predict(line), label="SVR gamma={}".format(gamma))
plt.plot(X[:, 0], y, "o", c="k")
plt.ylabel("회귀 출력")
plt.xlabel("입력 특성")
plt.legend(loc="best")
plt.show()

from sklearn.datasets import fetch_openml

# boston = fetch_openml(name='boston', version=1, as_frame=False)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

boston = fetch_openml(name="boston", version=1, as_frame=False)
boston.data.shape
boston.target.shape
X_train, X_test, y_train, y_test = train_test_split(
    boston.data, boston.target, random_state=0
)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

poly = PolynomialFeatures(degree=2).fit(X_train_scaled)
X_train_poly = poly.transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)
print("X_train.shape:", X_train.shape)
print("X_train_poly.shape:", X_train_poly.shape)

print("다항 특성 이름:\n", poly.get_feature_names_out())

from sklearn.linear_model import Ridge

ridge = Ridge().fit(X_train_scaled, y_train)
print("상호작용 특성이 없을 때 점수: {:.3f}".format(ridge.score(X_test_scaled, y_test)))
ridge = Ridge().fit(X_train_poly, y_train)
print("상호작용 특성이 있을 때 점수: {:.3f}".format(ridge.score(X_test_poly, y_test)))

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100).fit(X_train_scaled, y_train)
print("상호작용 특성이 없을 때 점수: {:.3f}".format(rf.score(X_test_scaled, y_test)))
rf = RandomForestRegressor(n_estimators=100).fit(X_train_poly, y_train)
print("상호작용 특성이 있을 때 점수: {:.3f}".format(rf.score(X_test_poly, y_test)))

rnd = np.random.RandomState(0)
X_org = rnd.normal(size=(1000, 3))
w = rnd.normal(size=3)

X = rnd.poisson(10 * np.exp(X_org))
y = np.dot(X_org, w)

print(X[:10, 0])

print("특성 출현 횟수:\n", np.bincount(X[:, 0]))

plt.xlim(0, 160)
plt.ylim(0, 70)
bins = np.bincount(X[:, 0])
plt.bar(range(len(bins)), bins, color="grey")
plt.ylabel("출현 횟수")
plt.xlabel("값")
plt.show()

from sklearn.linear_model import Ridge

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
score = Ridge().fit(X_train, y_train).score(X_test, y_test)
print("테스트 점수: {:.3f}".format(score))

X_train_log = np.log(X_train + 1)
X_test_log = np.log(X_test + 1)

plt.hist(X_train_log[:, 0], bins=25, color="gray")
plt.ylabel("출현 횟수")
plt.xlabel("값")
plt.show()

score = Ridge().fit(X_train_log, y_train).score(X_test_log, y_test)
print("테스트 점수: {:.3f}".format(score))

from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
# 고정된 난수를 발생시킵니다
rng = np.random.RandomState(42)
noise = rng.normal(size=(len(cancer.data), 50))
# 데이터에 노이즈 특성을 추가합니다
# 처음 30개는 원본 특성이고 다음 50개는 노이즈입니다
X_w_noise = np.hstack([cancer.data, noise])

X_train, X_test, y_train, y_test = train_test_split(
    X_w_noise, cancer.target, random_state=0, test_size=0.5
)
# f_classif(기본값)와 SelectPercentile을 사용하여特성의 50%를 선택합니다
select = SelectPercentile(score_func=f_classif, percentile=50)
select.fit(X_train, y_train)
# 훈련 세트에 적용합니다
X_train_selected = select.transform(X_train)

print("X_train.shape:", X_train.shape)
print("X_train_selected.shape:", X_train_selected.shape)

mask = select.get_support()
print(mask)
# True는 검은색, False는 흰색으로 마스킹합니다
plt.matshow(mask.reshape(1, -1), cmap="gray_r")
plt.xlabel("특성 번호")
plt.show()

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

select = SelectFromModel(
    RandomForestClassifier(n_estimators=100, random_state=42), threshold="median"
)
select.fit(X_train, y_train)
X_train_l1 = select.transform(X_train)
print("X_train.shape:", X_train.shape)
print("X_train_l1.shape:", X_train_l1.shape)

mask = select.get_support()
# True는 검은색, False는 흰색으로 마스킹합니다
plt.matshow(mask.reshape(1, -1), cmap="gray_r")
plt.xlabel("특성 번호")
plt.show()

X_test_l1 = select.transform(X_test)
score = (
    RandomForestClassifier(n_estimators=100, random_state=42)
    .fit(X_train_l1, y_train)
    .score(X_test_l1, y_test)
)
print("테스트 점수: {:.3f}".format(score))
