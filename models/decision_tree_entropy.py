import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def run(X_train, X_test, y_train, y_test):
    model = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Matplotlib Tree Visualization
    plt.figure(figsize=(20, 10))
    plot_tree(model, filled=True ,class_names=["Negative", "Positive"])
    plt.title("Decision Tree (Entropy)")
    plt.show()
  
    return "Random Forest (Entropy) .", y_pred