#%% # Set the Keras backend to TensorFlow
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras

#%% # Import necessary packages
import random
import shutil
import matplotlib.pyplot as plt
from keras import layers, applications, optimizers, callbacks
import tensorflow as tf
import numpy as np
import datetime
from tensorflow.keras.models import load_model

# Import custom helper functions
from helper_function import count_files_in_directory, split_files_three

#%% # Count the number of positive and negative image files in the dataset
positive_file_count = count_files_in_directory("datasets/concrete_crack_datasets/Positive")
negative_file_count = count_files_in_directory("datasets/concrete_crack_datasets/Negative")
print("Positive file count:", positive_file_count)
print("Negative file count:", negative_file_count)

#%% # Define dataset paths
DATASET_PATH = "datasets/concrete_crack_datasets"
DATA_SPLIT_PATH = "datasets/concrete_crack_split_datasets"

#%% # Split positive dataset into training, validation, and test subsets
src_dir_positive = DATASET_PATH + "/Positive"
des_dir_train_positive = DATA_SPLIT_PATH + "/train/Positive"
des_dir_valid_positive = DATA_SPLIT_PATH + "/valid/Positive"
des_dir_test_positive = DATA_SPLIT_PATH + "/test/Positive"
split_files_three(src_dir_positive, des_dir_train_positive, des_dir_valid_positive, des_dir_test_positive, split_ratio=[0.7, 0.15, 0.15])

#%% # Split negative dataset into training, validation, and test subsets
src_dir_negative = DATASET_PATH + "/Negative"
des_dir_train_negative = DATA_SPLIT_PATH + "/train/Negative"
des_dir_valid_negative = DATA_SPLIT_PATH + "/valid/Negative"
des_dir_test_negative = DATA_SPLIT_PATH + "/test/Negative"
split_files_three(src_dir_negative, des_dir_train_negative, des_dir_valid_negative, des_dir_test_negative, split_ratio=[0.7, 0.15, 0.15])

#%% # Remove the original dataset directory to save space
shutil.rmtree(r'datasets\concrete_crack_datasets')

#%% # Define paths for training, validation, and test datasets
train_path = DATA_SPLIT_PATH + "/train"
test_path = DATA_SPLIT_PATH + "/test"
valid_path = DATA_SPLIT_PATH + "/valid"

#%% # Create TensorFlow datasets for training, testing, and validation
BATCH_SIZE = 10
IMG_SIZE = (32, 32)
train_dataset = keras.utils.image_dataset_from_directory(train_path, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)
test_dataset = keras.utils.image_dataset_from_directory(test_path, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)
valid_dataset = keras.utils.image_dataset_from_directory(valid_path, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)

#%% # Display class names and a sample batch of images from the training dataset
class_names = train_dataset.class_names
batch = train_dataset.take(1)
for images, label in batch:
    plt.figure(figsize=(10, 10))
    for index in range(9):
        plt.subplot(3, 3, index + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[index].numpy().astype('uint8'))
        plt.title(class_names[label[index]])
plt.show()

#%% # Define a data augmentation pipeline
data_augmentation = keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.4),
    layers.RandomTranslation(0.2, 0.2),
    layers.RandomZoom(0.2)
])

#%% # Visualize augmented images for a sample image
for images, label in train_dataset.take(1):
    first_image = images[0]
    plt.figure()
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        augmented_image = data_augmentation(tf.expand_dims(first_image, axis=0))
        plt.imshow(augmented_image[0] / 255.0)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
plt.show()

#%% # Define the input shape for the model
IMG_SHAPE = IMG_SIZE + (3,)

#%% # Load EfficientNetV2B0 base model pre-trained on ImageNet
preprocess_input_efnv2 = applications.efficientnet_v2.preprocess_input
base_model_efnv2 = applications.EfficientNetV2B0(include_top=False, weights='imagenet', input_shape=IMG_SHAPE)
base_model_efnv2.trainable = False  # Freeze the base model
base_model_efnv2.summary()

#%% # Define additional layers for classification
global_avg_efnv2 = layers.GlobalAveragePooling2D()
output_layer_efnv2 = layers.Dense(len(class_names), activation='softmax')

#%% # Build the complete model
inputs_efnv2 = keras.Input(shape=IMG_SHAPE)
efnv2 = data_augmentation(inputs_efnv2)
efnv2 = preprocess_input_efnv2(efnv2)
efnv2 = base_model_efnv2(efnv2)
efnv2 = global_avg_efnv2(efnv2)
outputs_efnv2 = output_layer_efnv2(efnv2)
model_efnv2 = keras.Model(inputs=inputs_efnv2, outputs=outputs_efnv2)
model_efnv2.summary()

#%% Compile the model with an Adam optimizer
optimizer = optimizers.Adam(learning_rate=1e-3)
model_efnv2.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#%% Plot and save the model architecture
keras.utils.plot_model(model_efnv2, to_file="static/efficientnet_v2_b0.png", show_shapes=True)

#%% Set up TensorBoard for monitoring training
log_dir_initial = "logs/initial-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir_initial, histogram_freq=1)

#%% Train the model for initial training phase
EPOCHS_EFNV2 = 2
model_efnv2.fit(train_dataset, validation_data=valid_dataset, epochs=EPOCHS_EFNV2, callbacks=[tensorboard_callback])

#%% Print the TensorBoard command to view logs
print(f"Run `tensorboard --logdir={log_dir_initial}` to view logs in TensorBoard.")

#%% Fine-tune the model by unfreezing selective layers
# (A) Unfreeze the entire base model
base_model_efnv2.trainable = True
# (B) Freeze the earlier layers inside the base model
print(len(base_model_efnv2.layers))
fine_tune_at = 150 # for medium sized dataset
for layer in base_model_efnv2.layers[:fine_tune_at]:
    layer.trainable = False
for layer in base_model_efnv2.layers[fine_tune_at:]:
    layer.trainable = True
base_model_efnv2.summary()

#%% Compile the model with a lower learning rate for fine-tuning
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
model_efnv2.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#%% Set up fine-tuning logs and callbacks
log_dir_tuned = "logs/fine_tuned-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir_tuned, histogram_freq=1)

#%% Early stopping to prevent overfitting
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',       # Monitor validation loss
    patience=3,                    # Wait for 3 epochs without improvement
    min_delta=0.01,                # Minimum improvement of 1% in accuracy to continue training
    restore_best_weights=True,    # Restore the best weights after early stopping
    verbose=1
)

#%% Fine-tune the model
fine_tuned_epoch = 6
total_epoch = EPOCHS_EFNV2 + fine_tuned_epoch

#%% Start fine-tuning from where the initial training stopped
model_efnv2.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=total_epoch, 
    callbacks=[tensorboard_callback, early_stopping]
)

#%% Print the TensorBoard command to view fine-tuning logs
print(f"Run `tensorboard --logdir={log_dir_tuned}` to view logs in TensorBoard.")

#%% Save the trained model
os.makedirs("models", exist_ok=True)
model_efnv2.save("models/efficientnet_v2_b0.h5")

#%% Load the saved model for evaluation
loaded_model = load_model('models/efficientnet_v2_b0.h5')
loaded_model.summary()

#%% Evaluate the model on test and validation datasets
test_loss, test_accuracy = loaded_model.evaluate(test_dataset)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

val_loss, val_accuracy = loaded_model.evaluate(valid_dataset)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")

#%% Generate predictions for a batch of test images
for image_batch, label_batch in test_dataset.take(1):
    predictions = np.argmax(loaded_model.predict(image_batch), axis=1)
    predicted_class = [class_names[x] for x in predictions]
print(predicted_class)

#%% Visualize predictions and their corresponding true labels
plt.figure(figsize=(15, 15))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image_batch[i].numpy().astype('uint8'))
    plt.title(f"Prediction: {predicted_class[i]}, Label: {class_names[label_batch[i]]}")
plt.show()

# %%
