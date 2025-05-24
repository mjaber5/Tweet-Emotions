from sklearn.neural_network import MLPClassifier


def run(X_train, X_test, y_train, y_test):
    model = MLPClassifier(max_iter=300)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return "Neural Network",  y_pred
