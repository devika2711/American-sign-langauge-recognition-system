import pandas as pd
import numpy as np
import os
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator


train_data = pd.read_csv('sign_mnist_train.csv')
test_data = pd.read_csv('sign_mnist_test.csv')


train_labels = train_data['label'].values
train_images = train_data.drop('label', axis=1).values
test_labels = test_data['label'].values
test_images = test_data.drop('label', axis=1).values


train_labels = np.array([label - 1 if label > 9 else label for label in train_labels])
test_labels = np.array([label - 1 if label > 9 else label for label in test_labels])


train_images = train_images / 255.0
test_images = test_images / 255.0


train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)


train_labels = to_categorical(train_labels, num_classes=24)
test_labels = to_categorical(test_labels, num_classes=24)

# Create a CNN model
model = Sequential([
    Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=2),
    BatchNormalization(),
    Conv2D(64, kernel_size=3, activation='relu'),
    MaxPooling2D(pool_size=2),
    BatchNormalization(),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(24, activation='softmax')  # 24 classes for the letters A-Z (excluding J and Z)
])


def lr_scheduler(epoch):
    if epoch < 10:
        return 0.001
    else:
        return 0.0001

lr_callback = LearningRateScheduler(lr_scheduler)

# Compile the model with the Adam optimizer and a lower initial learning rate
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Implement early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Save the best model during training based on validation accuracy
model_checkpoint = ModelCheckpoint('best_sign_language_model.h5', save_best_only=True, 
                                   monitor='val_accuracy', mode='max', verbose=1)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Train the model with data augmentation
model.fit(datagen.flow(train_images, train_labels, batch_size=64), 
          epochs=50, 
          validation_data=(test_images, test_labels), 
          callbacks=[early_stopping, model_checkpoint, lr_callback])

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# Save images from the test dataset to the folder
if not os.path.exists('test_images'):
    os.makedirs('test_images')

for i in range(len(test_images)):
    image = test_images[i].reshape(28, 28) * 255.0  # Rescale to 0-255
    image = Image.fromarray(image.astype(np.uint8))
    label = test_labels[i]
    image.save(f'test_images/image_{i}_label_{label}.png')

print("Test images saved successfully in the 'test_images' folder.")
