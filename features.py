from utils import *
from sklearn.feature_selection import SelectKBest
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import statistics

X_train, y_train, X_val, y_val = load_data()
X_train, X_val = scale_data(X_train, X_val)

# train decision tree classifier
decf = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)

# show feature importances in graph
selector = SelectFromModel(decf, prefit=True)
feature_importance = decf.feature_importances_
plt.bar(range(0,128), feature_importance)
plt.xlabel("feature index")
plt.ylabel("Gini importance")
plt.axhline(y=statistics.mean(feature_importance), label='default threshold', c='r')
plt.legend()
plt.show()

# adjust threshold to get optimal f1 score
scores = []
for i in range(27, 40):
    selector = SelectFromModel(decf,  threshold=-np.inf, max_features=i)
    clf = Pipeline([
        ('feature_selection', selector),
        ('classification', RandomForestClassifier())
    ])
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_val,y_val))
plt.plot(range(27,40), scores)
plt.xlabel("number of features")
plt.ylabel("accuracy")
plt.show()

# select features by tree based gini 
MAX_FEATURE_LIMIT = 35
selector = SelectFromModel(decf, prefit=True, threshold=-np.inf, max_features=35)
print("selected feature indices by gini importance:")


