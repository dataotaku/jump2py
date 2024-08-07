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

import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import platform

# 시스템에 따라 폰트 설정
if platform.system() == "Windows":
    # Windows의 경우
    font_name = font_manager.FontProperties(
        fname="c:/Windows/Fonts/malgun.ttf"
    ).get_name()
    rc("font", family=font_name)
elif platform.system() == "Darwin":
    # Mac의 경우
    rc("font", family="AppleGothic")
else:
    # Linux의 경우
    rc("font", family="NanumGothic")

# 음수 기호가 깨지는 문제 해결
rc("axes", unicode_minus=False)
