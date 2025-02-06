import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Step 1: Load the preprocessed data
df = pd.read_csv('./processed_data_urls.csv')

# Ensure that the URLs are in list format (if they're in string format, eval to turn it into a list)
df['urls'] = df['urls'].apply(eval)

# Check the number of rows and the first few rows
print(f"Dataset size: {df.shape[0]}")
print(df.head())

# Step 2: Flatten the list of URLs to create one row per URL
# We'll explode the list of URLs into individual rows
df_exploded = df.explode('urls')

# Check the first few rows after exploding
print(f"Exploded dataset size: {df_exploded.shape[0]}")
print(df_exploded.head())

# Step 3: Define the features (X) and target labels (y)
X = df_exploded['urls']  # Use individual URLs as features
y = df_exploded['labels']  # Use the corresponding labels (phishing or not)

# Step 4: Split the data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Vectorize the URLs using TF-IDF (convert URLs into numerical features)
tfidf = TfidfVectorizer(stop_words='english', max_features=100)

# Fit and transform the training data
X_train_tfidf = tfidf.fit_transform(X_train)

# Transform the test data
X_test_tfidf = tfidf.transform(X_test)

# Step 6: Train using Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_tfidf, y_train)

# Step 7: Train using XGBoost (with CUDA support for GPU acceleration)
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    random_state=42,
    tree_method='hist',  # Use hist method
    device='cuda',       # Specify GPU device (use 'cpu' for CPU-only)
    eval_metric='logloss'  # Logloss for binary classification
)
xgb_model.fit(X_train_tfidf, y_train)

# Step 8: Save the trained models and the TF-IDF vectorizer for later use
joblib.dump(rf_model, 'rf_model.pkl')  # Save Random Forest model
joblib.dump(xgb_model, 'xgb_model.pkl')  # Save XGBoost model
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')  # Save the TF-IDF vectorizer

print("Training complete and models saved.")
