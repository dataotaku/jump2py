import numpy as np
from graphviz import Digraph
from scipy import sparse
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
import mglearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# import mglearn
# import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


x = np.array([[1, 2, 3], [4, 5, 6]])
print("x:\n", x)


# 대각선 원소는 1이고 나머지는 0인 2차원 NumPy 배열을 만듭니다.
eye = np.eye(4)
print("NumPy 배열:\n", eye)

# NumPy 배열을 CSR 포맷의 SciPy 희박 행렬로 변환합니다.
# 0이 아닌 원소만 저장됩니다.
sparse_matrix = sparse.csr_matrix(eye)
print("\nSciPy의 CSR 행렬:\n", sparse_matrix)

# Graphviz 패키지 설치 여부 확인 및 버전 확인
# 새로운 방향성 그래프 생성
dot = Digraph()

# 노드 추가
dot.node("A", "Node A")
dot.node("B", "Node B")
dot.node("C", "Node C")

# 엣지 추가
dot.edges(["AB", "BC"])
dot.edge("A", "C", constraint="false")

# 그래프를 파일로 저장하고 렌더링
dot.render("output/graph", format="png", view=True)

# sparse.coo_matrix() 함수는 COO 포맷을 이용하여 희소 행렬을 생성합니다.
data = np.ones(4)
row_indices = np.arange(4)
col_indices = np.arange(4)
eye_coo = sparse.coo_matrix((data, (row_indices, col_indices)))
print("\nCOO 표현:\n", eye_coo)

x = np.linspace(-10, 10, 100)
y = np.sin(x)
plt.plot(x, y, marker="x")
plt.show()

data = {
    "Name": ["John", "Anna", "Peter", "Linda"],
    "Location": ["New York", "Paris", "Berlin", "London"],
    "Age": [24, 13, 53, 33],
}
data_pandas = pd.DataFrame(data)
display(data_pandas)

data_pandas[data_pandas.Age > 30]


# 데이터셋 생성
X, y = make_blobs(centers=2, random_state=42)

# 데이터셋 시각화
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.title("Sample Data")
plt.show()

# 데이터셋 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# KNN 모델 생성 및 학습
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# 모델 시각화
mglearn.plots.plot_2d_separator(knn, X, fill=True, alpha=0.4)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.title("KNN Classifier")
plt.show()

sklearn.__version__

iris_dataset = load_iris()
print("iris_dataset key: \n{}".format(iris_dataset.keys()))

print(iris_dataset["DESCR"][:193] + "\n...")
print(iris_dataset["target_names"])
type(iris_dataset)
print("Target names: {}".format(iris_dataset["target_names"]))

print("Feature names: \n{}".format(iris_dataset["feature_names"]))

print("Type of data: {}".format(type(iris_dataset["data"])))
print("Shape of data: {}".format(iris_dataset["data"].shape))

print("First five columns of data:\n{}".format(iris_dataset["data"][:5]))

print("Type of target: {}".format(type(iris_dataset["target"])))
print("Shape of target: {}".format(iris_dataset["target"].shape))

print("Target:\n{}".format(iris_dataset["target"]))

iris_dataset["target_names"]

X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset["data"], iris_dataset["target"], random_state=0
)

print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))

print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))

iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)

pd.plotting.scatter_matrix(
    iris_dataframe,
    c=y_train,
    figsize=(15, 15),
    marker="o",
    hist_kwds={"bins": 20},
    s=60,
    alpha=0.8,
    cmap=mglearn.cm3,
    diagonal="kde",
)
# 산점도 행렬을 그립니다.

plt.show()

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape: {}".format(X_new.shape))

prediction = knn.predict(X_new)
print("Prediction: {}".format(prediction))
print("Predicted target name: {}".format(iris_dataset["target_names"][prediction]))

y_pred = knn.predict(X_test)
print("Test set predictions:\n{}".format(y_pred))

print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))

# 예제 2.3.1 예제에 사용할 데이터 셋 생성
X, y = mglearn.datasets.make_forge()
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(["Class 0", "Class 1"], loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
print("X.shape: {}".format(X.shape))
plt.show()

# 예제 2.3.2 wave 데이터셋 생성
X, y = mglearn.datasets.make_wave(n_samples=40)
plt.plot(X, y, "o")
plt.ylim(-3, 3)
plt.xlabel("Feature")
plt.ylabel("Target")
plt.show()

# 예제 2.3.3 유방암 데이터셋 생성
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
print("cancer.keys(): {}".format(cancer.keys()))

print("Shape of cancer data: {}".format(cancer.data.shape))
print(
    "Sample counts per class:\n{}".format(
        {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}
    )
)

# 정수 배열
x = np.array([0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 8])

# 빈도수 계산
counts = np.bincount(x)
print(counts)

print(cancer.feature_names)

# 예제 2.3.4 보스턴 주택가격 데이터셋 생성
X, y = mglearn.datasets.load_extended_boston()

X.shape

# 예제 2.3.5 K-최근접 이웃
mglearn.plots.plot_knn_classification(n_neighbors=1)
plt.show()

mglearn.plots.plot_knn_classification(n_neighbors=3)
plt.show()

# 예제 2.3.6 K-최근접 이웃 알고리즘 적용
X, y = mglearn.datasets.make_forge()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)
print("Test set predictions: {}".format(clf.predict(X_test)))
print("Test set accuracy: {:.2f}".format(clf.score(X_test, y_test)))

# 예제 2.3.7 KNeighborsClassifier 분석
fig, axes = plt.subplots(1, 3, figsize=(10, 3))
for n_neighbors, ax in zip([1, 3, 9], axes):
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=0.4)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("{} neighbor(s)".format(n_neighbors))
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
axes[0].legend(loc=3)
plt.show()
