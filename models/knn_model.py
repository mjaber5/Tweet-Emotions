from sklearn.neighbors import KNeighborsClassifier

def run(X_train, X_test, y_train, y_test):
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return "K-Nearest Neighbors (KNN)", y_pred
