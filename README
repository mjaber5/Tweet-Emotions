# 💬 Emotion Prediction from Twitter Posts (Binary Classification)

This project implements multiple machine learning models to predict **user emotions** as **Positive** or **Negative** based on Twitter posts. The system uses a pre-labeled dataset and applies natural language processing (NLP) techniques to clean and vectorize the data before training several classifiers.

---

## 📁 Project Structure

```
  Emotion Prediction/
  ├── data/
  │   └── tweet_emotions.csv             # Raw dataset
  ├── models/
  │   ├── decision_tree_entropy.py       # Decision Tree (Entropy)
  │   ├── decision_tree_gini.py          # Decision Tree (Gini)
  │   ├── knn_model.py                   # K-Nearest Neighbors
  │   ├── svm_model.py                   # Support Vector Machine
  │   ├── random_forest.py               # Random Forest
  │   └── neural_network.py              # Neural Network
  ├── outputs/
  │   ├── tree_entropy_matplotlib.png    # Decision tree plot (matplotlib)
  │   └── tree_entropy_graphviz.png      # Decision tree plot (Graphviz)
  ├── util/
  │   └── preprocess.py                  # Preprocessing & TF-IDF feature extraction
  ├── main.py                            # Main script to execute all models
  └── README.md                          # Documentation
```

---

## 🧠 Machine Learning Models

Each model is implemented in its own file and evaluated using common classification metrics:

- ✅ **K-Nearest Neighbors (KNN)**
- ✅ **Support Vector Machine (SVM)**
- ✅ **Decision Tree (Gini & Entropy)**
- ✅ **Random Forest**
- ✅ **Neural Network (MLPClassifier)**

### Evaluation Metrics:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

---

## 🧹 Data Preprocessing Pipeline

To ensure high-quality input for the models, the following steps are applied:

1. **Lowercase all text**
2. **Remove usernames (`@user`)**
3. **Remove special characters and numbers**
4. **Remove extra spaces**
5. **TF-IDF vectorization (top 5000 features)**

### Emotion Mapping:

- **Positive**: `joy`, `love`, `enthusiasm` → `Label 1`
- **Negative**: `anger`, `hate`, `sadness`, `worry`, `boredom`, `empty` → `Label 0`

---

## 📊 Visualizations

The `decision_tree_entropy.py` script generates two types of decision tree visualizations:

- **Matplotlib-based Tree Plot**  
  Saved as: `outputs/tree_entropy_matplotlib.png`

- **Graphviz-based Tree Diagram**  
  Saved as: `outputs/tree_entropy_graphviz.png`

> These visuals help understand how the model makes decisions based on feature splits.

---

## 🚀 How to Run the Project

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

> Make sure Graphviz is installed for tree visualizations:  
> https://graphviz.org/download/

### 2. Run the Main Script

```bash
python main.py
```

This will automatically:

- Load and clean the dataset
- Train and evaluate each model
- Output metrics to the console
- Save tree visualizations

---

## 📈 Sample Output

```
--- Support Vector Machine (SVM) ---
Accuracy: 0.8507
Precision: 0.7696
Recall: 0.4092
F1_score: 0.5343

--- Neural Network ---
Accuracy: 0.8016
Precision: 0.5239
Recall: 0.5711
F1_score: 0.5465
...
```

---

## 🚧 Potential Improvements

- 🔍 Use advanced text embeddings (Word2Vec, BERT, etc.)
- 🧪 Hyperparameter tuning with GridSearchCV
- 🎯 Try class balancing (SMOTE, undersampling)
- 🌍 Extend to multi-class classification
- 🕸️ Deploy as an interactive web app using Flask or Streamlit

---

## 📚 Dataset Source

The dataset is a modified version of an emotion-labeled Twitter dataset. It was cleaned and converted into a binary format for this classification task.

---

## 👨‍💻 Author

**Your Name**  
This project is part of a Machine Learning university presentation.

> Feel free to fork, contribute, or use it for educational purposes!

---

## 📄 License

This repository is released under the **MIT License**.  
Free for personal and educational use.
