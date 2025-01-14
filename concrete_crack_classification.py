#%%
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
print(keras.backend.backend())

#%%
import random
import shutil
import matplotlib.pyplot as plt
from keras import layers, applications, optimizers, callbacks
import tensorflow as tf
import numpy as np
import datetime
from tensorflow.keras.models import load_model

from helper_function import count_files_in_directory, split_files_three
# %%
positive_file_count = count_files_in_directory("datasets/concrete_crack_datasets/Positive")
negative_file_count = count_files_in_directory("datasets/concrete_crack_datasets/Negative")
print("Positive file count:",positive_file_count)
print("Negative file count:",negative_file_count)
# %%
DATASET_PATH = "datasets/concrete_crack_datasets"
DATA_SPLIT_PATH = "datasets/concrete_crack_split_datasets"
# %%
# positive dataset
src_dir_positive = DATASET_PATH+"/Positive"
des_dir_train_positive = DATA_SPLIT_PATH+"/train/Positive"
des_dir_valid_positive = DATA_SPLIT_PATH+"/valid/Positive"
des_dir_test_positive = DATA_SPLIT_PATH+"/test/Positive"

split_files_three(src_dir_positive,des_dir_train_positive, des_dir_valid_positive, des_dir_test_positive, split_ratio=[0.7,0.15,0.15])
# %%
# negative dataset
src_dir_negative = DATASET_PATH+"/Negative"
des_dir_train_negative = DATA_SPLIT_PATH+"/train/Negative"
des_dir_valid_negative = DATA_SPLIT_PATH+"/valid/Negative"
des_dir_test_negative = DATA_SPLIT_PATH+"/test/Negative"

split_files_three(src_dir_negative,des_dir_train_negative, des_dir_valid_negative, des_dir_test_negative, split_ratio=[0.7,0.15,0.15])
# %%
shutil.rmtree(r'datasets\concrete_crack_datasets')
# %%
train_path = DATA_SPLIT_PATH+"/train"
test_path = DATA_SPLIT_PATH+"/test"
valid_path = DATA_SPLIT_PATH+"/valid"
# %%
BATCH_SIZE = 10
IMG_SIZE = (32,32)
train_dataset = keras.utils.image_dataset_from_directory(train_path,shuffle=True,batch_size=BATCH_SIZE,image_size=IMG_SIZE)
test_dataset = keras.utils.image_dataset_from_directory(test_path,shuffle=True,batch_size=BATCH_SIZE,image_size=IMG_SIZE)
valid_dataset = keras.utils.image_dataset_from_directory(valid_path,shuffle=True,batch_size=BATCH_SIZE,image_size=IMG_SIZE)
# %%
class_names = train_dataset.class_names
batch_1 = train_dataset.take(1)
for images,label in batch_1:
    plt.figure(figsize=(10,10))
    for index in range(9):
        plt.subplot(3,3,index+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[index].numpy().astype('uint8'))
        plt.title(class_names[label[index]])
plt.show()
# %%
data_augmentation = keras.Sequential()
data_augmentation.add(layers.RandomFlip('Horizontal'))
data_augmentation.add(layers.RandomRotation(0.4))
data_augmentation.add(layers.RandomTranslation(0.2,0.2))
data_augmentation.add(layers.RandomZoom(0.2))
# %%
for images, label in train_dataset.take(1):
    first_image = images[0]
    plt.figure()
    for i in range(9):
        plt.subplot(3,3,i+1)
        augmented_image = data_augmentation(tf.expand_dims(first_image,axis=0))
        plt.imshow(augmented_image[0] / 255.0)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
plt.show()
# %%
IMG_SHAPE = IMG_SIZE + (3,)
# %%
# --------EFFICIENTNETV2 B0 MODEL-------------
# %%
preprocess_input_efnv2 = applications.efficientnet_v2.preprocess_input
# %%
base_model_efnv2 = applications.EfficientNetV2B0(include_top=False, weights='imagenet', input_shape=IMG_SHAPE)
# Freeze the entire base model
base_model_efnv2.trainable = False
base_model_efnv2.summary()
# %%
global_avg_efnv2 = layers.GlobalAveragePooling2D()
output_layer_efnv2 = layers.Dense(len(class_names),activation='softmax')
# %%
# a. Begin with input
inputs_efnv2 = keras.Input(shape=IMG_SHAPE)
# b. Augmentation
efnv2 = data_augmentation(inputs_efnv2)
# c. Image preprocessing
efnv2 = preprocess_input_efnv2(efnv2)
# d. feature extractor
efnv2 = base_model_efnv2(efnv2)
# e. classification layers
efnv2 = global_avg_efnv2(efnv2)
outputs_efnv2 = output_layer_efnv2(efnv2)
# f. define the keras model out
model_efnv2 = keras.Model(inputs=inputs_efnv2,outputs=outputs_efnv2)
model_efnv2.summary()
# %%
optimizer = optimizers.Adam(learning_rate=1e-3)
model_efnv2.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
# %%
keras.utils.plot_model(model_efnv2, to_file="static/efficientnet_v2_b0.png", show_shapes=True)
# %%
log_dir_initial = "logs/initial-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir_initial, histogram_freq=1)
# %%
EPOCHS_EFNV2 = 5
model_efnv2.fit(train_dataset,validation_data=(valid_dataset),epochs=EPOCHS_EFNV2,callbacks=[tensorboard_callback])
# %%
print(f"Run `tensorboard --logdir={log_dir_initial}` to view logs in TensorBoard.")
# %%
# ---------------------Fine-Tuned Training--------------------
# %%
optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5)
model_efnv2.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
# %%
log_dir_tuned = "logs/fine_tuned-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir_tuned, histogram_freq=1)
# %%
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',       # Monitor validation loss
    patience=3,                    # Wait for 5 epochs without improvement
    min_delta=0.01,                # Minimum improvement of 1% in accuracy to continue training
    restore_best_weights=True     # Restore the best weights after early stopping
)
# %%
fine_tuned_epoch = 5
# %%
model_efnv2.fit(train_dataset,validation_data=(valid_dataset),epochs=fine_tuned_epoch,callbacks=[tensorboard_callback,early_stopping])
# %%
print(f"Run `tensorboard --logdir={log_dir_tuned}` to view logs in TensorBoard.")
# %%
os.makedirs("models", exist_ok=True)
model_efnv2.save("models/efficientnet_v2_b0.h5")
# %%
loaded_model = load_model('models/efficientnet_v2_b0.h5')
# Check the model summary to verify it loaded correctly
loaded_model.summary()
# %%
# Evaluate the model on the test dataset
test_loss, test_accuracy = loaded_model.evaluate(test_dataset)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
# %%
for image_batch, label_batch in test_dataset.take(1):
    predictions = np.argmax(loaded_model.predict(image_batch),axis=1)
    predicted_class = [class_names[x] for x in predictions]
print(predicted_class)
# %%
plt.figure(figsize=(15,15))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image_batch[i].numpy().astype('uint8'))
    plt.title(f"Prediction: {predicted_class[i]}, Label: {class_names[label_batch[i]]}")
plt.show()
# %%
