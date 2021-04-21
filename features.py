from utils import *
from sklearn.feature_selection import SelectKBest
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel

X_train, y_train, X_val, y_val = load_data()
X_train, X_val = scale_data(X_train, X_val)

# elected features by tree based chi2 
selector = SelectKBest(chi2, k=26)
X_train_new = selector.fit(X_train, y_train)
print("features selecected by chi2:")
print(selector.get_support(range(128)))

# selected features by tree based entropy 
clf = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)
feature_importance = clf.feature_importances_
selector = SelectFromModel(clf, prefit=True)
print("features selecected tree based entropy:")
print(selector.get_support(range(128)))