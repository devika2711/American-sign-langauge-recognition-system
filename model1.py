import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Load the dataset
train_data = pd.read_csv('sign_mnist_train.csv')
test_data = pd.read_csv('sign_mnist_test.csv')

# Extract labels and images
train_labels = train_data['label'].values
train_images = train_data.drop('label', axis=1).values
test_labels = test_data['label'].values
test_images = test_data.drop('label', axis=1).values

# Adjust labels - subtract 1 from labels greater than 9
train_labels = np.array([label - 1 if label > 9 else label for label in train_labels])
test_labels = np.array([label - 1 if label > 9 else label for label in test_labels])

# Normalize the images
train_images = train_images / 255.0
test_images = test_images / 255.0

# Reshape the data to fit the model
train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)

# One-hot encode the labels
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

# Compile the model with the Adam optimizer and a lower learning rate
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Implement early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Save the best model during training
model_checkpoint = ModelCheckpoint('best_sign_language_model.h5', save_best_only=True, verbose=1)

# Train the model with batch_size=64 and more epochs
model.fit(train_images, train_labels, epochs=30, batch_size=64, validation_data=(test_images, test_labels),
          callbacks=[early_stopping, model_checkpoint])

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

print("Model trained and evaluated successfully!")
