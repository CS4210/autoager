import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths for CSV files
train_data_path = r"C:\Users\solar\OneDrive\Documents\GitHub\autoager\CSV\Car Train.csv"
test_data_path = r"C:\Users\solar\OneDrive\Documents\GitHub\autoager\CSV\Car Test.csv"

# Check if CSV files exist
if not os.path.isfile(train_data_path):
    raise FileNotFoundError(f"Training data CSV not found: {train_data_path}")
if not os.path.isfile(test_data_path):
    raise FileNotFoundError(f"Test data CSV not found: {test_data_path}")

# Load the training and test data
train_df = pd.read_csv(train_data_path)
test_df = pd.read_csv(test_data_path)

# Debugging: Display the first few rows of the DataFrames and column names
print("Training DataFrame Head:\n", train_df.head())
print("Test DataFrame Head:\n", test_df.head())
print(f"Train DataFrame Columns: {train_df.columns}")
print(f"Test DataFrame Columns: {test_df.columns}")

# Rename columns - adjust based on your actual column names
# If 'Image ID' does not exist, use the actual column name for the image identifiers
train_df.rename(columns={'Pictures': 'image_path', 'Model': 'label'}, inplace=True)
test_df.rename(columns={'Pictures': 'image_path', 'Model': 'label'}, inplace=True)

# Debugging: Check columns after renaming
print(f"Train DataFrame Columns after rename: {train_df.columns}")
print(f"Test DataFrame Columns after rename: {test_df.columns}")

# Define image folders
train_images_folder = r"C:\Users\solar\OneDrive\Documents\GitHub\autoager\cars_train"
test_images_folder = r"C:\Users\solar\OneDrive\Documents\GitHub\autoager\cars_test"

# Add full paths to the DataFrame
train_df['image_path'] = train_df['image_path'].apply(lambda x: os.path.join(train_images_folder, f"{x}.jpg"))
test_df['image_path'] = test_df['image_path'].apply(lambda x: os.path.join(test_images_folder, f"{x}.jpg"))

# Drop rows where image_path is missing or invalid
train_df = train_df.dropna(subset=['image_path'])
test_df = test_df.dropna(subset=['image_path'])

# Verify file existence and filter invalid paths
train_df = train_df[train_df['image_path'].apply(os.path.isfile)]
test_df = test_df[test_df['image_path'].apply(os.path.isfile)]

# Debugging: Print the number of valid images
print(f"Valid training images: {len(train_df)}")
print(f"Valid test images: {len(test_df)}")

if train_df.empty or test_df.empty:
    raise ValueError("No valid image files found. Check your image paths and directory content.")
