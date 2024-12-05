import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DataEngineer:
    def __init__(self, image_dir, train_dir, test_dir, accident_dir, train_csv, test_csv, output_train_dir, output_val_dir, output_test_dir, output_accident_dir):
        # Initialize class variables with directory paths
        self.image_dir = image_dir
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.accident_dir = accident_dir
        self.train_csv = train_csv
        self.test_csv = test_csv
        self.output_train_dir = output_train_dir
        self.output_val_dir = output_val_dir
        self.output_test_dir = output_test_dir
        self.output_accident_dir = output_accident_dir

        # Create output directories if they don't exist
        os.makedirs(self.output_train_dir, exist_ok=True)
        os.makedirs(self.output_val_dir, exist_ok=True)
        os.makedirs(self.output_test_dir, exist_ok=True)
        os.makedirs(self.output_accident_dir, exist_ok=True)

    def load_images_and_metadata(self, csv_path, images_dir, output_dir):
        # Initialize lists to store images and labels
        images, labels = [], []
        if csv_path:
            # Load metadata from CSV file
            data = pd.read_csv(csv_path)
            for _, row in data.iterrows():
                img_path = os.path.join(images_dir, row['Pictures'])
                if os.path.exists(img_path):  # Check if image exists
                    image = cv2.imread(img_path)  # Read the image
                    if image is not None:  # Check if image was loaded successfully
                        image = cv2.resize(image, (224, 224))  # Resize image
                        images.append(image)
                        labels.append(row['Year'])  # Assuming 'Year' column contains labels
                        resized_image_path = os.path.join(output_dir, row['Pictures'])
                        cv2.imwrite(resized_image_path, image)  # Save resized image
                    else:
                        print(f"Warning: Could not load image {img_path}. Skipping.")
        else:
            # Process images without a CSV
            for img_file in os.listdir(images_dir):
                img_path = os.path.join(images_dir, img_file)
                if os.path.exists(img_path):  # Check if image exists
                    image = cv2.imread(img_path)  # Read the image
                    if image is not None:  # Check if image was loaded successfully
                        image = cv2.resize(image, (224, 224))  # Resize image
                        images.append(image)
                        resized_image_path = os.path.join(output_dir, img_file)
                        cv2.imwrite(resized_image_path, image)  # Save resized image
                    else:
                        print(f"Warning: Could not load image {img_path}. Skipping.")
        return np.array(images), np.array(labels) if labels else None

    def preprocess_images(self, images):
        # Normalize pixel values to the range [0, 1] for neural network input
        return images / 255.0

    def augment_images(self, images):
        # Augment images by flipping them horizontally
        augmented_images = []
        for image in images:
            flipped = cv2.flip(image, 1)  # Horizontal flip
            augmented_images.append(flipped)
        return np.array(augmented_images)

    def create_data_pipeline(self):
        # Load training data and resize images
        X_train_full, y_train_full = self.load_images_and_metadata(self.train_csv, self.train_dir, self.output_train_dir)
        
        # Split the training data into training and validation sets (80:20 split)
        X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)
        
        # Save validation images to output directory
        for i, val_image in enumerate(X_val):
            val_image_path = os.path.join(self.output_val_dir, f"val_image_{i}.jpg")
            cv2.imwrite(val_image_path, val_image)
        
        # Preprocess training and validation data
        X_train = self.preprocess_images(X_train)
        X_val = self.preprocess_images(X_val)

        # Augment training data (flipping images)
        X_train_augmented = self.augment_images(X_train)

        # Load and preprocess testing data
        X_test, y_test = self.load_images_and_metadata(self.test_csv, self.test_dir, self.output_test_dir)
        X_test = self.preprocess_images(X_test)

        # Load and preprocess accident images (without CSV)
        X_accident, _ = self.load_images_and_metadata(None, self.accident_dir, self.output_accident_dir)
        X_accident = self.preprocess_images(X_accident)

        return (X_train, y_train, X_train_augmented), (X_val, y_val), (X_test, y_test), X_accident

# Dynamically set paths relative to the script's location
base_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(base_dir, "Accident Images")
train_dir = os.path.join(base_dir, "Car Train Images")
test_dir = os.path.join(base_dir, "Car Test Images")
train_csv = os.path.join(base_dir, "CSV", "Car_Train.csv")
test_csv = os.path.join(base_dir, "CSV", "Car_Test.csv")

# Output directories for resized images
output_train_dir = os.path.join(base_dir, "Resized Train Images")
output_val_dir = os.path.join(base_dir, "Resized Validation Images")
output_test_dir = os.path.join(base_dir, "Resized Test Images")
output_accident_dir = os.path.join(base_dir, "Resized Accident Images")

# Instantiate and process data
data_engineer = DataEngineer(image_dir, train_dir, test_dir, image_dir, train_csv, test_csv, output_train_dir, output_val_dir, output_test_dir, output_accident_dir)
(train_data, train_labels, train_augmented), (val_data, val_labels), (test_data, test_labels), accident_data = data_engineer.create_data_pipeline()

# Print dataset shapes for verification
print(f"Train Data Shape: {train_data.shape}, Train Labels Shape: {train_labels.shape}")
print(f"Validation Data Shape: {val_data.shape}, Validation Labels Shape: {val_labels.shape}")
print(f"Test Data Shape: {test_data.shape}, Test Labels Shape: {test_labels.shape}")
print(f"Augmented Train Data Shape: {train_augmented.shape}")
print(f"Accident Data Shape: {accident_data.shape}")

# Confirm resized images saved
print(f"Resized train images saved in: {output_train_dir}")
print(f"Resized validation images saved in: {output_val_dir}")
print(f"Resized test images saved in: {output_test_dir}")
print(f"Resized accident images saved in: {output_accident_dir}")
