import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Define paths for Excel files
train_data_path = r"C:\Users\solar\OneDrive\Documents\GitHub\autoager\CSV\Car_Train.xlsx"
test_data_path = r"C:\Users\solar\OneDrive\Documents\GitHub\autoager\CSV\Car_Test.xlsx"

# Check if Excel files exist
if not os.path.isfile(train_data_path):
    raise FileNotFoundError(f"Training data Excel file not found: {train_data_path}")
if not os.path.isfile(test_data_path):
    raise FileNotFoundError(f"Test data Excel file not found: {test_data_path}")

# Load the training and test data from Excel
train_df = pd.read_excel(train_data_path)
test_df = pd.read_excel(test_data_path)

# Debugging: Display the first few rows of the DataFrames and column names
logging.info("Training DataFrame Head:\n%s", train_df.head())
logging.info("Test DataFrame Head:\n%s", test_df.head())
logging.info(f"Train DataFrame Columns: {train_df.columns}")
logging.info(f"Test DataFrame Columns: {test_df.columns}")

# Rename columns - adjust based on actual column names
train_df.rename(columns={'Pictures': 'image_path', 'Model': 'label'}, inplace=True)
test_df.rename(columns={'Pictures': 'image_path', 'Model': 'label'}, inplace=True)

# Debugging: Check columns after renaming
logging.info(f"Train DataFrame Columns after rename: {train_df.columns}")
logging.info(f"Test DataFrame Columns after rename: {test_df.columns}")

# Define image folders
train_images_folder = r"C:\Users\solar\OneDrive\Documents\GitHub\autoager\cars_train\Car Train Images"
test_images_folder = r"C:\Users\solar\OneDrive\Documents\GitHub\autoager\cars_test\Car Test Images"

# Helper function to construct valid image paths
def find_image_path(image_id, folder):
    """Construct full image path with supported extensions."""
    for ext in ['.jpg', '.png', '.jpeg']:
        path = os.path.join(folder, f"{image_id}{ext}")
        if os.path.isfile(path):
            return path
    return None

# Add full paths to the DataFrame
train_df['image_path'] = train_df['image_path'].apply(lambda x: find_image_path(x, train_images_folder))
test_df['image_path'] = test_df['image_path'].apply(lambda x: find_image_path(x, test_images_folder))

# Drop rows with missing or invalid image paths
train_df.dropna(subset=['image_path'], inplace=True)
test_df.dropna(subset=['image_path'], inplace=True)

# Validate file existence (already handled in find_image_path)
logging.info(f"Valid training images: {len(train_df)}")
logging.info(f"Valid test images: {len(test_df)}")

if train_df.empty or test_df.empty:
    raise ValueError("No valid image files found. Check your image paths and directory content.")

# Clean and validate labels
train_df['label'] = train_df['label'].str.strip()
test_df['label'] = test_df['label'].str.strip()

# Debugging: Display a few sample rows with paths
logging.info("Sample training data:\n%s", train_df[['image_path', 'label']].head())
logging.info("Sample test data:\n%s", test_df[['image_path', 'label']].head())
