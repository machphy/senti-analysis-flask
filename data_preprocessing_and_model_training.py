import nltk
from nltk.corpus import movie_reviews
import random
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Downloaddataset
nltk.download('movie_reviews')
nltk.download('punkt')
nltk.download('stopwords')

# Load dataset
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)

# Preprocess 
stopwords = nltk.corpus.stopwords.words('english')
def preprocess(document):
    words = [w.lower() for w in document if w.isalpha()]
    words = [w for w in words if w not in stopwords]
    return ' '.join(words)

documents = [(preprocess(doc), category) for doc, category in documents]
data = pd.DataFrame(documents, columns=['text', 'sentiment'])

# Split 
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['sentiment'], test_size=0.2, random_state=42)

# Create a pipeline with TF-IDF and Naive Bayes
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train model
model.fit(X_train, y_train)

# Evaluate  model
predicted = model.predict(X_test)
accuracy = accuracy_score(y_test, predicted)
print(f"Accuracy: {accuracy}")

# Sav model
joblib.dump(model, 'sentiment_model.pkl')
