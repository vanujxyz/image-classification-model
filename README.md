<h1>Image Classification Model - CIFAR-10 Dataset</h1>

<p>This project implements an image classification model using the CIFAR-10 dataset, which contains 60,000 32x32 color images across 10 different classes (e.g., airplane, dog, truck). The model is built using a Convolutional Neural Network (CNN) in TensorFlow/Keras.</p>

<h2>Steps Followed</h2>
<ol>
  <li><strong>Setup & Initialization:</strong>
    <p>I created a new directory, initialized a Git repository, and installed necessary libraries like TensorFlow and Matplotlib.</p>
  </li>
  <li><strong>Data Preprocessing:</strong>
    <p>Loaded the CIFAR-10 dataset using Keras, normalized the images, and one-hot encoded the labels.</p>
  </li>
  <li><strong>Building the Model:</strong>
    <p>Designed a CNN with Conv2D layers for feature extraction, MaxPooling for downsampling, and Dense layers for classification. Added Batch Normalization and Dropout to reduce overfitting.</p>
  </li>
  <li><strong>Training:</strong>
    <p>Compiled the model with the Adam optimizer and categorical crossentropy loss. Trained the model for 30 epochs, using early stopping to prevent overfitting.</p>
  </li>
  <li><strong>Evaluation:</strong>
    <p>Evaluated the model on the test set, computed accuracy, precision, recall, and F1-score to assess performance.</p>
  </li>
  <li><strong>Saving the Model:</strong>
    <p>Saved the trained model as <code>model.h5</code> for future use.</p>
  </li>
  <li><strong>Making Predictions:</strong>
    <p>Created a separate script (<code>self_image.py</code>) to load the saved model and make predictions on custom images.</p>
  </li>
</ol>

<h2>How to Run the Project</h2>
<p>1. <strong>Clone the Repository:</strong></p>
<pre><code>git clone https://github.com/your-username/image-classification-model.git
cd image-classification-model</code></pre>

<p>2. <strong>Install Dependencies:</strong></p>
<pre><code>pip install -r requirements.txt</code></pre>

<p>3. <strong>Train the Model:</strong><br> 
To train the model from scratch, run:</p>
<pre><code>python main.py</code></pre>

<p>4. <strong>Make Predictions:</strong><br> 
Use the <code>self_image.py</code> script to classify your own images:</p>
<pre><code>python self_image.py</code></pre>

<h2>Project Structure</h2>
<pre><code>
image-classification-model/
│
├── main.py                # Train and evaluate the model
├── self_image.py          # Make predictions on custom images
├── model.h5               # Saved trained model
├── requirements.txt       # Dependencies
└── README.md              # Project overview
</code></pre>

<h2>Dependencies</h2>
<ul>
  <li>Python 3.x</li>
  <li>TensorFlow 2.x</li>
  <li>Keras</li>
  <li>Matplotlib</li>
  <li>Numpy</li>
  <li>Scikit-learn</li>
</ul>

<p>Install all dependencies with:</p>
<pre><code>pip install -r requirements.txt</code></pre>
