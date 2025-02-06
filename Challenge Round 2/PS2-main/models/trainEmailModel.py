# Step 1: Install Dependencies (Run this first in the terminal if needed)
# pip install transformers datasets torch scikit-learn

import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.metrics import accuracy_score

# Step 2: Load the Processed Data (assuming your file is processed_data.csv)
df = pd.read_csv('./processed_data_2.csv')

# Check the first few rows of the dataset to ensure it's loaded correctly
print(df.head())

# Step 3: Prepare the Dataset (only body and labels are needed for training)
df = df[['body', 'labels']]  # Keep 'body' and 'labels' columns

# Step 4: Split into Training and Validation Sets (80% train, 20% validation)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Convert to Hugging Face Dataset format
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Check the data format
print(train_dataset[0])  # Check an example from the train dataset

# Step 5: Tokenize the Text Using BERT Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples['body'], padding="max_length", truncation=True, max_length=128)

# Apply tokenization to both the training and validation datasets
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Set the format of the datasets for PyTorch
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Step 6: Load Pre-trained BERT Model for Sequence Classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Move the model to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Step 7: Define Training Arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory for model checkpoints
    num_train_epochs=3,              # number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,                # log every 10 steps
    evaluation_strategy="epoch",     # evaluate after every epoch
    save_strategy="epoch",           # save checkpoint after every epoch
)

# Step 8: Define Evaluation Metric (Accuracy)
def compute_metrics(p):
    preds = p.predictions.argmax(axis=-1)
    return {'accuracy': accuracy_score(p.label_ids, preds)}

# Step 9: Initialize the Trainer
trainer = Trainer(
    model=model,                         # the model to train
    args=training_args,                  # training arguments
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset,            # evaluation dataset
    compute_metrics=compute_metrics      # evaluation metrics
)

# Step 10: Start Training
trainer.train()
