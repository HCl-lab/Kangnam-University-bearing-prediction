from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# 用字典存储机器学习模型
models = {
    # 'LR': LogisticRegression(),
    # 'SVM': SVC(),
    'RF': RandomForestClassifier(),
    # 'DT': DecisionTreeClassifier(),
    # # 'GBDT': GradientBoostingClassifier(verbose=1),
    # 'KNN': KNeighborsClassifier(),
    # 'NB': GaussianNB()
}



