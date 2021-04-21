from utils import *
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

chi2=[0,2,3,8,10,11,17,50,58,64,66,67,68,72,74,75,76,81,82,83,90,91,114,115,122,124]
entropy=[8,11,15,16,19,24,25,35,37,66,68,69,73,75,77,78,80,81,83,84,96,100,104,112,115,123]

X_train, y_train, X_val, y_val = load_data()
X_train, X_val = scale_data(X_train, X_val)

def run_classifier(classifier, my_X_train, my_y_train):
    clf = None
    if classifier == "SVC":
        clf = SVC().fit(my_X_train, my_y_train)
    elif classifier == "LogisticRegression":
        clf = LogisticRegression().fit(my_X_train, my_y_train)
    elif classifier == "AdaBoostClassifier":
        clf = AdaBoostClassifier(n_estimators=50).fit(my_X_train, my_y_train)
    elif classifier == "RandomForestClassifier":
        clf = RandomForestClassifier(max_depth=None).fit(my_X_train, my_y_train)
    elif classifier == "DecisionTreeClassifier":
        clf = DecisionTreeClassifier(random_state=0).fit(my_X_train, my_y_train)
    elif classifier == "MLPClassifier":
        clf = MLPClassifier().fit(my_X_train, my_y_train)
    elif classifier == "KNeighborsClassifier":
        clf = KNeighborsClassifier(n_neighbors=3).fit(my_X_train, my_y_train)
    return clf



classifiers = [
    "SVC",
    "LogisticRegression",
    "AdaBoostClassifier",
    "RandomForestClassifier",
    "KNeighborsClassifier",
    "DecisionTreeClassifier",
    "MLPClassifier",
]

#X_train_selected = feature_selected_chi2(X_train)
#X_val_selected = feature_selected_chi2(X_val)
X_train_selected = feature_selected(X_train, entropy)
X_val_selected = feature_selected(X_val, entropy)
for classifier in classifiers:
    clf = run_classifier(classifier, X_train, y_train)
    clf_new = run_classifier(classifier, X_train_selected, y_train)
    print(classifier, ": in training set")
    training_score = clf.score(X_train, y_train)
    print(training_score)
    print(classifier, ": in testing set")
    testing_score = clf.score(X_val, y_val)
    print(testing_score)
    print("after selected features")
    testing_score_new = clf_new.score(X_val_selected, y_val) 
    print("diff:" + str(testing_score_new - testing_score))
    continue


