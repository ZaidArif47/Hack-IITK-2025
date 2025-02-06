import pandas as pd
import re

# Function to parse and clean the message
def parse_and_clean_message(message):
    # Find the first empty line (headers end before this)
    body_start = message.find("\n\n")
    if body_start == -1:
        body_start = 0  # If no empty line found, treat the whole message as body
    
    # Extract the body content after the headers
    body = message[body_start:].strip()  # Remove leading/trailing spaces
    
    return body

# Function to extract features from each email message
def extract_features_from_message(message):
    # Clean the message to extract the body
    body = parse_and_clean_message(message)
    
    # Feature extraction:
    email_length = len(body)  # Length of the email body
    
    # Find all URLs in the body, if any
    urls = re.findall(r'http[s]?://[^\s]+', body)
    urls = urls if urls else None  # Store list of URLs, or None if no URLs found

    labels = 1 if any(keyword in body.lower() for keyword in ['urgent', 'emergency', 'verify', 'password', 'click here', 'now', 'immediately', 'free', 'increase', 'money', 'gain', 'guaranteed']) else 0
    
    # Extract sender and recipient emails using regex
    sender_email = re.search(r'From:\s*([^\s]+@[^\s]+)', message)
    recipient_email = re.search(r'To:\s*([^\s]+@[^\s]+)', message)
    
    sender_email = sender_email.group(1) if sender_email else None
    recipient_email = recipient_email.group(1) if recipient_email else None
    
    # Extract the Message-ID
    message_id = re.search(r'Message-ID:\s*<([^>]+)>', message)
    message_id = message_id.group(1) if message_id else None
    
    # Generalized extraction of attachments from both plain text and HTML formats
    attachment_matches_text = re.findall(r'- ([^ ]+\.\w+)', body)  # For text-based attachments
    attachment_matches_html = re.findall(r'filename="([^"]+\.\w+)"', body)  # For HTML-based attachments
    attachment_matches_img_tags = re.findall(r'src="([^"]+\.\w+)"', body)  # Capture images in HTML

    # Combine all found attachments into a single list, ensuring uniqueness
    attachments = list(set(attachment_matches_text + attachment_matches_html + attachment_matches_img_tags))

    return message_id, sender_email, recipient_email, body, email_length, urls, attachments, labels

def preprocessEmail(emailFile, outputFile='processed_data.csv'):
    # Load the original CSV dataset
    df = pd.read_csv(emailFile)

    # Initialize the list for storing processed data
    processed_data = []

    # Iterate over the rows of the dataset and extract features
    for index, row in df.iterrows():
        message = row['message']
        
        # Extract features from the message
        message_id, sender_email, recipient_email, body, email_length, urls, attachments, labels = extract_features_from_message(message)
        
        # Add extracted data to the processed list
        processed_data.append([message_id, sender_email, recipient_email, body, email_length, urls, attachments, labels])

    # Create a new DataFrame with the extracted features
    processed_df = pd.DataFrame(processed_data, columns=['message_id', 'sender_email', 'recipient_email', 'body', 'email_length', 'urls', 'attachments', 'labels'])

    # Save the processed data to a new CSV
    processed_df.to_csv(outputFile, index=False)

    print("Data preprocessing complete. Saved to " + str(outputFile) + ".")

if "__name__" == "__main__":
    emailFile = './datasets/emails.csv'
    preprocessEmail(emailFile)