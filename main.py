# Imports
import tensorflow as tf
from tensorflow.keras.datasets import cifar10 #for the dataset
from tensorflow.keras.utils import to_categorical ##for the dataset
import matplotlib.pyplot as plt #for the dataset
from tensorflow.keras.models import Sequential, load_model #for training the cnn
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization #for training the cnn
from tensorflow.keras.callbacks import EarlyStopping #for training the cnn
from sklearn.metrics import classification_report #for testing
import os

#loading model
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalising the dataset
x_train, x_test = x_train / 255.0, x_test / 255.0

# converting labels to encoding format
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# testing the dataset
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_train[i])
    plt.title(class_names[y_train[i].argmax()])
    plt.axis('off')
plt.show()

#model training function
def build_model():
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        # Third Convolutional Block
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        # Flattening and connecting the layers
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')  # 10 output classes
    ])
    
    # Compiling model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# loading model for save so that it doesnt train again and again
model_path = 'image_classification_model.h5'

# Check if the model file exists
if os.path.exists(model_path):
    # Load the pre-trained model
    model = load_model(model_path)
    print("Loaded pre-trained model.")
else:
    # Build and train the model if it doesn't exist
    model = build_model()
    
    # Define early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Train the model
    history = model.fit(x_train, y_train, epochs=30, batch_size=64,
                        validation_data=(x_test, y_test), callbacks=[early_stop])
    
    # Save the trained model
    model.save(model_path)
    print(f"Model trained and saved to {model_path}")

#TESTING
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'Test Accuracy: {test_acc * 100:.2f}%')
print(f'Test Loss: {test_loss:.4f}')

# Calculate Precision, Recall, and F1-Score
y_pred = model.predict(x_test)
y_pred_classes = y_pred.argmax(axis=1)
y_true = y_test.argmax(axis=1)

# Print classification report
print(classification_report(y_true, y_pred_classes))
