import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Step 1: Load the trained model and tokenizer
model = BertForSequenceClassification.from_pretrained('./models/emailcheck')  # Path to your model checkpoint
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # BERT tokenizer

# Step 2: Define the device (GPU if available, otherwise CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Step 3: Define a function to preprocess and make predictions
def predict_phishing(email_text):
    # Ensure the email text is a valid string and not NaN
    if not isinstance(email_text, str) or not email_text.strip():
        return None  # Return None if the text is empty, NaN, or invalid

    # Tokenize the email text
    inputs = tokenizer(email_text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    
    # Move the input tensors to the appropriate device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Make prediction
    with torch.no_grad():  # No need to track gradients for inference
        model.eval()  # Set the model to evaluation mode
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)  # Get the predicted class

    # 0 -> Legitimate email, 1 -> Phishing email
    return predictions.item()

# Step 4: Load your Enron dataset
df = pd.read_csv('./datasets/testEmails.csv')

# Step 5: Combine 'subject' and 'body' for each email (assuming these are the column names)
df['text'] = df['subject'] + " " + df['body']

# Step 6: Get the true labels
true_labels = df['label'].tolist()

# Step 7: Make predictions on the dataset
predictions = []
valid_true_labels = []  # To store the valid true labels
for email_text, label in zip(df['text'], true_labels):
    # Check if the email text is a valid string and not NaN
    if pd.isna(email_text) or not isinstance(email_text, str) or not email_text.strip():
        continue  # Skip empty, NaN, or invalid email texts
    
    result = predict_phishing(email_text)
    if result is not None:  # Only append valid predictions
        predictions.append(result)
        valid_true_labels.append(label)

# Step 8: Compute metrics if there are any valid predictions
if predictions:
    accuracy = accuracy_score(valid_true_labels, predictions)
    precision = precision_score(valid_true_labels, predictions)
    recall = recall_score(valid_true_labels, predictions)
    f1 = f1_score(valid_true_labels, predictions)

    # Step 9: Print the evaluation metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
else:
    print("No valid email texts to process.")


"""
OUTPUT ->
Accuracy: 0.5547
Precision: 0.5171
Recall: 0.6709
F1 Score: 0.5840
"""