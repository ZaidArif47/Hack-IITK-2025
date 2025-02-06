# Phishing Email and URL Detection System

This project implements a system for detecting phishing emails and malicious URLs using machine learning models. The system includes preprocessing of email data, prediction of phishing emails using a BERT-based model, and detection of malicious URLs using Random Forest and XGBoost models.

We have included a demonstration video named **demo-video.mp4** to showcase the system in action.

## Setup and Usage

### Installation

#### Step 0: Download our trained BERT model emailcheck from the Google Drive Link
1. Download from https://drive.google.com/file/d/1fAAvcSxDD7f6ekIo1bzc8vIc1Dp7glkY/view?usp=sharing
2. Unzip the emailcheck file
3. Place emailcheck folder under the models folder


#### Step 1: Set up a virtual environment

##### Windows:
1. Open Command Prompt.
2. Navigate to the project folder.
3. Run the following commands to set up a virtual environment:
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

##### macOS/Linux:
1. Open Terminal.
2. Navigate to the project folder.
3. Run the following commands to set up a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

#### Step 2: Install dependencies

1. Once the virtual environment is activated, install the required dependencies using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

#### Step 3: Run the application

1. Ensure all required models are in the appropriate directories:
   - BERT model for email classification: `./models/emailcheck/`
   - Random Forest and XGBoost models for URL classification: `./models/urlcheck/`

2. Run the main application file to start the GUI:
    ```bash
    python main.py
    ```

3. Use the "Upload Email File" button to select a CSV file containing email data.

4. The system will preprocess the data, predict phishing emails, and detect malicious URLs.

5. Results will be displayed in the text widget, showing the classification for each email and URL.


## Files Overview

### main.py
This is the main application file that provides a graphical user interface (GUI) for the phishing detection system.

**Key Features:**
- File upload functionality for email data (CSV format)
- Preprocessing of uploaded email data
- Phishing email prediction
- Malicious URL detection
- Progress bar for processing status
- Display of results in a text widget

### preprocessing.py
This file contains functions for preprocessing email data.

**Key Functions:**
- `parse_and_clean_message(message)`: Extracts the body content from an email message
- `extract_features_from_message(message)`: Extracts various features from an email message, including sender, recipient, body, URLs, and attachments
- `preprocessEmail(emailFile, outputFile)`: Processes a CSV file containing email data and saves the extracted features to a new CSV file

### model_test_email.py
This file implements the phishing email prediction using a BERT-based model.

**Key Features:**
- Loads a pre-trained BERT model for sequence classification
- Defines a `predict_phishing(email_text)` function that classifies an email as "phishing" or "safe content"

### model_test_urls.py
This file contains functions for predicting malicious URLs using Random Forest and XGBoost models.

**Key Features:**
- Loads pre-trained Random Forest and XGBoost models
- Implements a `predict_urls(url_list)` function that classifies a list of URLs as malicious or not malicious

### getEmailPrecision.py
This file is used to evaluate the performance of the email classification model.

**Key Features:**
- Loads the trained BERT model for email classification
- Processes a test dataset of emails
- Calculates and prints evaluation metrics (accuracy, precision, recall, F1 score)

### getUrlPrecision.py
This file is used to evaluate the performance of the URL classification models.

**Key Features:**
- Loads the trained Random Forest and XGBoost models for URL classification
- Processes a test dataset of URLs
- Prints predictions for both models
- Calculates and prints evaluation metrics for both models

### helper.py
This file contains utility functions used in the project.

**Key Functions:**
- `convert_to_list(url_str)`: Converts a string representation of a list to an actual list, used for processing URL data

### trainEmailModel.py
This script is used to train the BERT-based model for email classification. This script was used to train the model that was subsequently used in the main application for prediction tasks. It is not part of the main execution flow but is crucial for preparing the machine learning model used in the phishing detection system.

**Key Features:**
- Loads and preprocesses the email dataset
- Splits data into training and validation sets
- Tokenizes text using BERT tokenizer
- Initializes a pre-trained BERT model for sequence classification
- Defines training arguments and evaluation metrics
- Trains the model using the Hugging Face Trainer

### trainUrlModel.py
This script is used to train the Random Forest and XGBoost models for URL classification. This script was used to train the model that was subsequently used in the main application for prediction tasks. It is not part of the main execution flow but is crucial for preparing the machine learning model used in the phishing detection system.

**Key Features:**
- Loads and preprocesses the URL dataset
- Flattens the list of URLs to create one row per URL
- Vectorizes URLs using TF-IDF
- Trains a Random Forest Classifier
- Trains an XGBoost Classifier with CUDA support for GPU acceleration
- Saves the trained models and TF-IDF vectorizer for later use


## Model Information

- Email classification uses a fine-tuned BERT model for sequence classification.
- URL classification uses both Random Forest and XGBoost models.
- The models should be trained separately and saved in the appropriate directories before running the application or evaluation scripts.

### Model Evaluation

To evaluate the performance of the models separately from the main application:

1. Run `getEmailPrecision.py` to assess the email classification model's performance:
    ```bash
    python getEmailPrecision.py
    ```

2. Run `getUrlPrecision.py` to evaluate the URL classification models' performance:
    ```bash
    python getUrlPrecision.py
    ```

These scripts will process test datasets and output various evaluation metrics.

## Model Performance

### Random Forest Model Performance

|              | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| 0            | 1.00      | 0.07   | 0.13     | 29      |
| 1            | 0.48      | 1.00   | 0.65     | 25      |
| **Accuracy** |           |        | 0.50     | 54      |
| Macro Avg    | 0.74      | 0.53   | 0.39     | 54      |
| Weighted Avg | 0.76      | 0.50   | 0.37     | 54      |

### XGBoost Model Performance

|              | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| 0            | 0.67      | 1.00   | 0.81     | 29      |
| 1            | 1.00      | 0.44   | 0.61     | 25      |
| **Accuracy** |           |        | 0.74     | 54      |
| Macro Avg    | 0.84      | 0.72   | 0.71     | 54      |
| Weighted Avg | 0.83      | 0.74   | 0.72     | 54      |

### BERT Model Performance

- **Accuracy:** 0.5547
- **Precision:** 0.5171
- **Recall:** 0.6709
- **F1 Score:** 0.5840

## Notes

- The system processes emails in batches, updating the progress bar as it goes.
- The final results display the count of phishing and safe emails detected.
- The evaluation scripts (`getEmailPrecision.py` and `getUrlPrecision.py`) are run separately from the main application to assess model performance on test datasets.
- We have included a demonstration video named **demo-video.mp4** to showcase the system in action.