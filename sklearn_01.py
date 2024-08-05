import numpy as np
from graphviz import Digraph
from scipy import sparse

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
