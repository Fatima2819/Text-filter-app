import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from joblib import dump
import os

# Get paths relative to THIS script's location
script_dir = os.path.dirname(__file__)  # Path to training/ folder
csv_path = os.path.join(script_dir, '..', 'toxic_comments.csv')
model_dir = os.path.join(script_dir, '..', 'model')

# Load data
df = pd.read_csv(csv_path)
X = df['comment']
y = df['label']

# Train model
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)
model = MultinomialNB()
model.fit(X_vec, y)

# Save to model/ folder
dump(model, os.path.join(model_dir, 'model.joblib'))
dump(vectorizer, os.path.join(model_dir, 'vectorizer.joblib'))
print("Model trained and saved!")