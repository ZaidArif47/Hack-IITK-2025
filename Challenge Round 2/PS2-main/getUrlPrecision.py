
import joblib
import pandas as pd
from sklearn.metrics import classification_report

def predictUrl(urlFilePath):
    # Load the saved models and TF-IDF vectorizer
    rf_model = joblib.load('./models/urlcheck/rf_model.pkl')
    xgb_model = joblib.load('./models/urlcheck/xgb_model.pkl')
    tfidf = joblib.load('./models/urlcheck/tfidf_vectorizer.pkl')

    # Load your test dataset (e.g., a CSV file with 'urls' column and 'labels' for ground truth)
    test_df = pd.read_csv(urlFilePath)  # Update to the path of your test dataset

    # Ensure URLs are in list format and convert from string to list if necessary
    test_df['urls'] = test_df['urls'].apply(eval)

    # Exploding the 'urls' column to create one row per URL
    test_exploded = test_df.explode('urls')

    # Prepare features (URLs) and labels
    X_test = test_exploded['urls']  # Extract URLs as features
    y_test = test_exploded['labels']  # Extract corresponding labels

    # Transform the test data (URLs) into TF-IDF features using the same vectorizer as the training set
    X_test_tfidf = tfidf.transform(X_test)

    # Predict using the Random Forest model
    y_pred_rf = rf_model.predict(X_test_tfidf)

    # Predict using the XGBoost model
    y_pred_xgb = xgb_model.predict(X_test_tfidf)

    return X_test, y_test, X_test_tfidf, y_pred_rf, y_pred_xgb

urlFilePath = './datasets/processed_data_urls3.csv'
X_test, y_test, X_test_tfidf, y_pred_rf, y_pred_xgb = predictUrl(urlFilePath)

# Print URLs with their predictions for Random Forest
print("Random Forest Model Predictions:")
for url, pred in zip(X_test, y_pred_rf):
    result = "Malicious" if pred == 1 else "Not Malicious"
    print(f"URL: {url}\nPrediction: {result}\n")

# Print URLs with their predictions for XGBoost
print("\nXGBoost Model Predictions:")
for url, pred in zip(X_test, y_pred_xgb):
    result = "Malicious" if pred == 1 else "Not Malicious"
    print(f"URL: {url}\nPrediction: {result}\n")

# Evaluate the Random Forest model performance
print("Random Forest Model Performance:")
print(classification_report(y_test, y_pred_rf))

# Evaluate the XGBoost model performance
print("XGBoost Model Performance:")
print(classification_report(y_test, y_pred_xgb))