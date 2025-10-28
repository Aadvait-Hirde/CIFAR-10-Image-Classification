# CIFAR 10 Image Classification

This Python project is a real-time image classification system built for the CIFAR-10 dataset using TensorFlow and OpenCV. It focuses on training a Convolutional Neural Network (CNN) to classify images into ten categories such as airplanes, cars, and animals. The CIFAR-10 dataset is loaded, preprocessed, and split into training and testing datasets. Images are normalized to enhance the model’s performance during training and evaluation.

The CNN model is designed with a series of convolutional layers for feature extraction, followed by batch normalization and dropout layers to improve stability and prevent overfitting. Dense layers with softmax activation handle the final classification task. The model is optimized using the Adam optimizer and employs advanced training techniques such as early stopping and learning rate reduction to ensure efficient convergence. Once trained, the model achieves high accuracy, and its performance is evaluated using various metrics, including confusion matrices, per-class accuracy, ROC curves, and precision-recall curves. These metrics provide detailed insights into how the model performs across all categories.

A key feature of the project is its integration of deep learning with computer vision for real-time inference. Using OpenCV, video frames are captured from a webcam, and a specific region of interest (ROI) is processed and resized to match the CNN’s input size. The model predicts the class of the object in the ROI and overlays the predicted label and confidence score on the live video feed. Users can toggle inference mode by clicking on the OpenCV window and exit the system with a simple keypress.

This project delivers a complete system that integrates a trained CNN model into a real-time application. Its flexible design makes it easy to adapt to other datasets and classification tasks, making it a practical tool for various use cases. By combining clear evaluation methods, real-time interaction, and a simple workflow, this project showcases how deep learning can be effectively used for real-world image classification tasks.

<p align="center">
  <img src="assets/demo.png" alt="Demo Preview" width="full">
</p>
