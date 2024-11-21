import os
import pandas as pd
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

# Rename columns to match the required format
train_df.rename(columns={'Pictures': 'image_path', 'Model': 'label'}, inplace=True)
test_df.rename(columns={'Pictures': 'image_path', 'Model': 'label'}, inplace=True)

# Debugging: Check columns after renaming
logging.info(f"Train DataFrame Columns after rename: {train_df.columns}")
logging.info(f"Test DataFrame Columns after rename: {test_df.columns}")

# Define image folders
train_images_folder = r"C:\Users\solar\OneDrive\Documents\GitHub\autoager\cars_train\Car Train Images"
test_images_folder = r"C:\Users\solar\OneDrive\Documents\GitHub\autoager\cars_test\Car Test Images"

# Normalize image names in the folders
def normalize_image_names(folder):
    """Normalize file names in the folder to lowercase and stripped of spaces."""
    for file in os.listdir(folder):
        normalized_name = file.strip().lower()
        os.rename(os.path.join(folder, file), os.path.join(folder, normalized_name))

normalize_image_names(train_images_folder)
normalize_image_names(test_images_folder)

# Helper function to construct valid image paths
def find_image_path(image_id, folder):
    """Construct full image path with supported extensions."""
    image_id = image_id.strip().lower()  # Normalize to lowercase
    for ext in ['.jpg', '.png', '.jpeg']:
        path = os.path.join(folder, f"{image_id}{ext}")
        if os.path.isfile(path):
            return path
    logging.debug(f"File not found: {os.path.join(folder, image_id)}")
    return None

# Add full paths to the DataFrame
train_df['image_path'] = train_df['image_path'].apply(lambda x: find_image_path(x, train_images_folder))
test_df['image_path'] = test_df['image_path'].apply(lambda x: find_image_path(x, test_images_folder))

# Log missing images for debugging
missing_train_images = train_df[train_df['image_path'].isna()]
missing_test_images = test_df[test_df['image_path'].isna()]
logging.info(f"Missing train images: {len(missing_train_images)}")
logging.info(f"Missing test images: {len(missing_test_images)}")

if not missing_train_images.empty:
    logging.info("Missing train image details:\n%s", missing_train_images)
if not missing_test_images.empty:
    logging.info("Missing test image details:\n%s", missing_test_images)

# Drop rows with missing or invalid image paths
train_df.dropna(subset=['image_path'], inplace=True)
test_df.dropna(subset=['image_path'], inplace=True)

# Validate remaining image paths
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

# Ready for further processing (e.g., loading images, training a model)
