import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from preprocessing import preprocessEmail
from model_test_email import predict_phishing
from model_test_urls import predict_urls
from helper import convert_to_list
import pandas as pd
import threading  # To run long tasks in the background without freezing the UI

# Function to handle file upload
def upload_file():
    emailFile = filedialog.askopenfilename(title="Select Email File", filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
    if emailFile:
        file_label.config(text=emailFile)  # Update label to show uploaded file path
        upload_status_label.config(text="File uploaded successfully!")  # Confirmation message
        start_processing_thread(emailFile)  # Start processing in a separate thread

# Function to process the email file and show output
def process_data(emailFile):
    phishing, safe = 0, 0
    result_text.delete(1.0, tk.END)  # Clear previous output

    # Preprocess the data
    preprocessEmail(emailFile)
    
    try:
        df = pd.read_csv('./processed_data.csv')
    except FileNotFoundError:
        messagebox.showerror("Error", "Processed data file not found after preprocessing.")
        return

    # Total number of emails to process for progress calculation
    total_emails = len(df)
    progress_step = total_emails if total_emails > 0 else 1  # Avoid division by zero

    for index, row in df.iterrows():
        body = row['body']
        urls = row['urls']

        result = predict_phishing(body)
        result_text.insert(tk.END, f"Phishing result for email {index + 1}: {result}\n")

        rfUrlTest = {}
        xgbUrlTest = {}

        if pd.isna(urls):
            result_text.insert(tk.END, "No URLs\n")
        else:
            _, xgbUrlTest = predict_urls(convert_to_list(urls))

            if any(value is False for value in xgbUrlTest.values()):
                result_text.insert(tk.END, "Phishing URLs\n")
            else:
                result_text.insert(tk.END, "Safe URLs\n")

        if result == "phishing" or any(value is False for value in xgbUrlTest.values()):
            phishing += 1
        else: 
            safe += 1

        # Update progress bar
        progress_var.set((index + 1) * 100 / progress_step)
        root.update_idletasks()  # Update the UI

    # Final results
    result_text.insert(tk.END, f"\nPhishing count: {phishing}\nSafe count: {safe}\n")
    upload_status_label.config(text=f"Processing complete: Phishing={phishing}, Safe={safe}")  # Show final counts

# Function to start processing in a separate thread to prevent blocking the UI
def start_processing_thread(emailFile):
    processing_thread = threading.Thread(target=process_data, args=(emailFile,))
    processing_thread.daemon = True
    processing_thread.start()

# Setting up the main window
root = tk.Tk()
root.title("Phishing Email Prediction")

# File upload button
upload_button = tk.Button(root, text="Upload Email File", command=upload_file)
upload_button.pack(pady=20)

# Label to show the file path after upload
file_label = tk.Label(root, text="No file uploaded", width=50, anchor="w")
file_label.pack(pady=5)

# Label for upload confirmation
upload_status_label = tk.Label(root, text="", width=50, anchor="w", fg="green")
upload_status_label.pack(pady=5)

# Progress bar (taskbar)
progress_var = tk.DoubleVar()
progressbar = ttk.Progressbar(root, variable=progress_var, maximum=100, length=400)
progressbar.pack(pady=20)

# Text widget to display the result outputs
result_text = tk.Text(root, width=80, height=20)
result_text.pack(pady=20)

# Start the Tkinter event loop
root.mainloop()
