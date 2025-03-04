import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load and preprocess the image
def load_and_preprocess_image(img_path):
    # Load the image from the file path and resize it to match CIFAR-10 input shape
    img = image.load_img(img_path, target_size=(32, 32))  # Resize to 32x32
    img_array = image.img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image
    return img_array

# Predict the class of the image
def predict_image(model, img_path):
    img_array = load_and_preprocess_image(img_path)  # Load and preprocess the image
    predictions = model.predict(img_array)  # Get model prediction
    predicted_class = np.argmax(predictions)  # Get the class with the highest probability
    return predicted_class, predictions[0][predicted_class]

# Main function to run the prediction
if __name__ == "__main__":
    # Load your trained model (make sure your model is saved as 'model.h5' or use your model file)
    model = tf.keras.models.load_model('image_classification_model.h5')

    # Class names corresponding to CIFAR-10 dataset (adjust if needed)
    class_names = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    
    # Provide the image path you want to test
    img_path = 'sports-car-parked-in-barcelona.jpg'  # Replace with the path to your image

    # Get the prediction
    predicted_class, confidence = predict_image(model, img_path)

    # Display the image and predicted class with confidence
    plt.imshow(image.load_img(img_path))
    plt.title(f"Predicted Class: {class_names[predicted_class]} with confidence {confidence:.2f}")
    plt.axis('off')
    plt.show()
