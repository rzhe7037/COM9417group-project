from utils import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# loading data from csv file and MinMaxScale data
X_train, y_train, X_val, y_val = load_data()

# feature selection based on gini importance
X_train = feature_selected(X_train, gini)
X_val = feature_selected(X_val, gini)

# search optimal n_estimater
scores = []
MAX_ESTIMATER = 50
for i in range(1, MAX_ESTIMATER):
    rfc = RandomForestClassifier(random_state=0, n_estimators=i, n_jobs=-1)
    rfc_crs = cross_val_score(rfc, X_train, y_train, cv=10).mean()
    scores.append(rfc_crs)
print(max(scores), scores.index(max(scores)))
plt.plot(range(MAX_ESTIMATER), scores)
plt.xlabel("n_estimator")
plt.ylabel("accuracy")
plt.show()

# search optimal min_samples_leaf
scores = []
MAX_LEAVE_NODE = 20
for i in range(1, MAX_LEAVE_NODE):
    rfc = RandomForestClassifier(random_state=0, min_samples_leaf=i, n_jobs=-1)
    rfc_crs = cross_val_score(rfc, X_train, y_train, cv=10).mean()
    scores.append(rfc_crs)
print(max(scores), scores.index(max(scores)))
plt.plot(range(MAX_LEAVE_NODE), scores)
plt.xlabel("min_samples_leaf")
plt.ylabel("accuracy")
plt.show()

# GridSearch parameter setting
forest = RandomForestClassifier(
    random_state=0, n_estimators=20, min_samples_leaf=1, bootstrap=False
)
param_grid = {
    "criterion": ["gini", "gini"],
    "min_samples_split": range(2, 20),
    "max_depth": [10, 20, 30, 40, 50, 60, 70, 80],
    "max_features": ["auto", "sqrt", "log2"],
    "bootstrap": [True, False],
}
search_cv = GridSearchCV(forest, param_grid, cv=10, n_jobs=-1, scoring="r2")
search_cv.fit(X_train, y_train)
print("Best params:  \n")
print(search_cv.best_params_)

