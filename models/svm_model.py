from sklearn.svm import SVC

def run(X_train, X_test, y_train, y_test):
    model = SVC()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return "Support Vector Machine (SVM)", y_pred
