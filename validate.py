from utils import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


# loading data from csv file and MinMaxScale data
X_train, y_train, X_val, y_val = load_data()

# feature selection based on gini importance
X_train = feature_selected(X_train, gini)
X_val = feature_selected(X_val, gini)

# fit to the training data with the optimal parameter set
RFC = RandomForestClassifier(random_state=0, n_estimators= 20, min_samples_split= 5, min_samples_leaf= 1, max_features= "auto", max_depth= 60, bootstrap= False)
RFC.fit(X_train,y_train)
y_pred = RFC.predict(X_val)

# evaluate performance of the model
print("confusion matrix:")
print(confusion_matrix(y_val, RFC.predict(X_val),labels=[1,2,3,4,5,6]))

print("f1 score:")
print(RFC.score(X_val,y_val))

print("precision score:")
print(precision_score(y_val, y_pred, average='macro'))

print("recall score:")
print(recall_score(y_val, y_pred, average='macro'))

