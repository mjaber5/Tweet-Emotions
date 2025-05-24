from sklearn.ensemble import RandomForestClassifier

def run(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return "Random Forest", y_pred
