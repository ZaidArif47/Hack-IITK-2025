import joblib
import pandas as pd

def predict_urls(url_list):
    # Load the saved models and TF-IDF vectorizer
    rf_model = joblib.load('./models/urlcheck/rf_model.pkl')
    xgb_model = joblib.load('./models/urlcheck/xgb_model.pkl')
    tfidf = joblib.load('./models/urlcheck/tfidf_vectorizer.pkl')

    # Convert the list to a DataFrame
    test_df = pd.DataFrame({'urls': url_list})

    # Transform the URLs into TF-IDF features using the same vectorizer as the training set
    X_test_tfidf = tfidf.transform(test_df['urls'])

    # Predict using the Random Forest model
    y_pred_rf = rf_model.predict(X_test_tfidf)

    # Predict using the XGBoost model
    y_pred_xgb = xgb_model.predict(X_test_tfidf)

    rfUrlTest = {}
    xgbUrlTest = {}
    # Print results row by row
    print("URL Predictions:\n")
    for url, rf_pred, xgb_pred in zip(url_list, y_pred_rf, y_pred_xgb):
        rf_result = "Malicious" if rf_pred == 1 else "Not Malicious"
        xgb_result = "Malicious" if xgb_pred == 1 else "Not Malicious"
        # print(f"URL: {url}\nRandom Forest Prediction: {rf_result}\nXGBoost Prediction: {xgb_result}\n{'-'*50}")
        # rfUrlTest[url] = rf_pred == 1  # True if Malicious, False otherwise
        # xgbUrlTest[url] = xgb_pred == 1  # True if Malicious, False otherwise   
        if url not in rfUrlTest:
            if rf_pred == 1:
                rfUrlTest[url] = True
            else: 
                rfUrlTest[url] = False
        if url not in xgbUrlTest:
            if xgb_pred == 1:
                xgbUrlTest[url] = True
            else: 
                xgbUrlTest[url] = False

    return rfUrlTest, xgbUrlTest

if __name__ == "__main__":
    # Example usage
    url_list = ["http://shockwave1.m0.net/m/s.asp?H430297053X351632", "http://shockwave1.m0.net/m/s.asp?H430297053X351633", "http://living.com/shopping/list/list.jhtml?type=2011&sale=valentines"]
    rfUrlTest, xgbUrlTest = predict_urls(url_list)

    print(rfUrlTest)
    print(xgbUrlTest)

