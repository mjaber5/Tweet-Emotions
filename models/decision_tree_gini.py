from sklearn.tree import DecisionTreeClassifier

def run(X_train, X_test, y_train, y_test):
    model = DecisionTreeClassifier(criterion='gini')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return "Decision Tree (Gini)", y_pred
