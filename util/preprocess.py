import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    if pd.isnull(text):
        return ""
    
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)     # remove URLs
    text = re.sub(r'@\w+|#\w+', '', text)                   # remove mentions and hashtags
    text = re.sub(r'[^a-z\s]', '', text)                    # keep only letters
    text = re.sub(r'\s+', ' ', text).strip()                # normalize spaces
    
    # Stopword removal and stemming
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    
    return ' '.join(words)

def load_and_preprocess(path):
    df = pd.read_csv(path)

    # Check required columns
    if 'content' not in df.columns or 'sentiment' not in df.columns:
        raise ValueError("Expected columns 'content' and 'sentiment' not found.")

    # Drop rows with missing values
    df.dropna(subset=['content', 'sentiment'], inplace=True)

    # Define emotion categories
    positive_emotions = ['joy', 'love', 'enthusiasm']
    negative_emotions = ['anger', 'hate', 'sadness', 'worry', 'boredom', 'empty']
    df = df[df['sentiment'].isin(positive_emotions + negative_emotions)]

    # Binary label assignment
    df['label'] = df['sentiment'].apply(lambda x: 1 if x in positive_emotions else 0)

    # Clean the text
    df['clean_content'] = df['content'].apply(clean_text)

    # TF-IDF with unigrams + bigrams
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=7000)
    X = vectorizer.fit_transform(df['clean_content'])
    y = df['label']

    # Optional: handle class imbalance with SMOTE
    sm = SMOTE(random_state=42)
    X_resampled, y_resampled = sm.fit_resample(X, y)

    # Shuffle for good measure
    X_resampled, y_resampled = shuffle(X_resampled, y_resampled, random_state=42)

    # Train/test split
    return train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
