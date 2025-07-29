import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# --- Define file path ---
# The script will now look for the CSV file in the same directory
file_path = "twitter_training.csv"

# --- Load the training data correctly ---
# The CSV file has no header, so we must provide column names
try:
    training = pd.read_csv(
        file_path,
        header=None,
        names=['id', 'company', 'sentiment', 'post_content']
    )
except FileNotFoundError:
    print(f"Error: The file was not found at '{file_path}'")
    print("Please make sure 'twitter_training.csv' is in the same folder as this script.")
    exit()


# 1. Initialize the LabelEncoder
label_encoder = LabelEncoder()

# 2. Fit the encoder on your training data's 'sentiment' column
# This learns the mapping from "Positive", "Negative", etc. to 0, 1, 2, 3
label_encoder.fit(training['sentiment'])

# 3. Save the fitted encoder to a file using pickle
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("✅ label_encoder.pkl saved successfully.")

# --- You can now use this encoder in your main training script ---
# training['sentiment_encoded'] = label_encoder.transform(training['sentiment'])
