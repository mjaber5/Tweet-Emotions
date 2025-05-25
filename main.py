from util.preprocess import load_and_preprocess
from models import decision_tree_gini, decision_tree_entropy, svm_model, random_forest, neural_network, knn_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load preprocessed data
X_train, X_test, y_train, y_test = load_and_preprocess('data/tweet_emotions.csv')

# Model list
models = [
    knn_model,
    svm_model,           
    decision_tree_entropy,   
    decision_tree_gini,
    random_forest,
    neural_network
]

# Loop through each model
for model in models:
    name, y_pred = model.run(X_train, X_test, y_train, y_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n--- {name} ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1_score: {f1:.4f}\n")
