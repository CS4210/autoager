import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths for training and test data
train_data_path = r"C:\Users\solar\OneDrive\Documents\GitHub\autoager\Car Train.csv"
test_data_path = r"C:\Users\solar\OneDrive\Documents\GitHub\autoager\Car Test.csv"

# Check if CSV files exist and provide feedback if they do not
if not os.path.isfile(train_data_path):
    print(f"Training data CSV not found at: {train_data_path}")
    raise FileNotFoundError("Training data CSV path is incorrect or does not exist.")
if not os.path.isfile(test_data_path):
    print(f"Test data CSV not found at: {test_data_path}")
    raise FileNotFoundError("Test data CSV path is incorrect or does not exist.")

# Load the training and test data from CSV
train_df = pd.read_csv(train_data_path)
test_df = pd.read_csv(test_data_path)

# Debugging: Print the DataFrames and their columns
print("Training DataFrame:")
print(train_df.head())  # Print the first few rows of the DataFrame
print("Columns in Training DataFrame:", train_df.columns.tolist())  # Print the column names

# Rename columns to standardize format if necessary
train_df.rename(columns={'Model': 'label', 'ID Image': 'image_path'}, inplace=True)
test_df.rename(columns={'Model': 'label', 'ID Image': 'image_path'}, inplace=True)

# Define image folders for training and test images
train_images_folder = r"C:\Users\solar\OneDrive\Desktop\4210 Images\train_images"
test_images_folder = r"C:\Users\solar\OneDrive\Desktop\4210 Images\test_images"

# Construct the full image path for each image in the DataFrame
train_df['image_path'] = train_df['image_path'].apply(lambda x: os.path.join(train_images_folder, f"{x}.jpg"))
test_df['image_path'] = test_df['image_path'].apply(lambda x: os.path.join(test_images_folder, f"{x}.jpg"))

# Image dimensions and batch size
img_width, img_height = 224, 224
batch_size = 32

# Set up data augmentation for the training set
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,           # Normalization
    rotation_range=20,             # Rotate images
    width_shift_range=0.1,         # Horizontal shift
    height_shift_range=0.1,        # Vertical shift
    horizontal_flip=True,          # Flip images horizontally
    zoom_range=0.1                 # Zoom for augmentation
)

# Only rescale the test set without augmentation
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

# Create a flow for training data from DataFrame
train_data = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='image_path',
    y_col='label',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

# Create a flow for test data from DataFrame
test_data = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='image_path',
    y_col='label',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

# Verify that images are found and classes are detected
print(f"Training data found: {train_data.samples} images in {len(train_data.class_indices)} classes.")
print(f"Test data found: {test_data.samples} images in {len(test_data.class_indices)} classes.")

if train_data.samples == 0 or test_data.samples == 0:
    raise ValueError("No images found in the specified DataFrames. Check file paths and DataFrame content.")

# Verify if GPU is available and set the strategy for processing
strategy = tf.distribute.MirroredStrategy() if tf.config.list_physical_devices('GPU') else tf.distribute.get_strategy()
print(f"Using strategy: {strategy}")

# Wrap model creation within the GPU strategy scope
with strategy.scope():
    # Define a simple CNN model for prediction
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(img_width, img_height, 3)),  # Input layer
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(train_data.num_classes, activation='softmax')  # Output layer
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_data, validation_data=test_data, epochs=10)

# Example prediction on a batch of test images
test_images, test_labels = next(test_data)  # Get a batch of test images and labels
predictions = model.predict(test_images)

# Output the predictions for each image in the batch
for i, pred in enumerate(predictions):
    print(f"Image {i} predicted class: {pred.argmax()}, actual class: {test_labels[i].argmax()}")
