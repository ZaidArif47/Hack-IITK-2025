from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd

# Step 3: Define a function to preprocess and make predictions
def predict_phishing(email_text):
    # Tokenize the email text

    # Step 1: Load the trained model and tokenizer
    model = BertForSequenceClassification.from_pretrained('./models/emailcheck')  # Path to your model checkpoint
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # BERT tokenizer

    # Step 2: Define the device (GPU if available, otherwise CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    inputs = tokenizer(email_text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    
    # Move the input tensors to the appropriate device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Make prediction
    with torch.no_grad():  # No need to track gradients for inference
        model.eval()  # Set the model to evaluation mode
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)  # Get the predicted class

    # 0 -> Legitimate email, 1 -> Phishing email
    if predictions.item() == 1:
        return "phishing"
    else:
        return "safe content"

# Step 4: Test with a sample email
# email_sample = "Urgent! Your account has been compromised. Please click the link to reset your password immediately."

# result = predict_phishing(email_sample)
# print(result)
if __name__ == "__main__":
    df = pd.read_csv('./processed_data.csv')

    # print("=================")
    # print(df[['body']])
    # print("=================")
    # for email in df['body']:  # Use df['body'] instead of df[['body']]
    #     print(email)
    print("=================")
    for email in df['body']:
        if email:
            result = predict_phishing(email)
            print(f"Email: {email}\nPrediction: {result}\n")
    print("=================")

