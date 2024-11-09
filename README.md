# Drowsiness Detection System

This repository provides a Drowsiness Detection System designed to enhance driver safety by detecting signs of drowsiness in real-time. The project leverages a Convolutional Neural Network (CNN) for classifying eye states as "drowsy" or "non-drowsy."

## Repository Structure

- **Drowsiness_Detection.ipynb**: This Jupyter Notebook covers the entire model training process, including data preprocessing, model architecture design (based on a modified VGG16), training, and evaluation. The model is optimized for high accuracy and generalization on live data.
  
- **live_webcam_detection.py**: A Python script for real-time drowsiness detection using the webcam. The script uses OpenCV to capture video, detect faces and eyes, and classify eye states by passing the detected eye regions to the trained model. If drowsiness is detected, the system can alert the user in real time.

## Features

- **Model Architecture**: A modified VGG16 model tailored for binary classification (drowsy vs. non-drowsy).
- **Real-Time Detection**: Uses OpenCV for real-time face and eye detection, enabling seamless integration with webcam input for live drowsiness monitoring.
- **High Accuracy**: Achieves 99% accuracy on the validation set and generalizes well to live data, ensuring robust performance in real-world settings.

## Requirements

- Python 3.x
- Jupyter Notebook
- TensorFlow and Keras
- OpenCV

## Usage

1. **Model Training**: Open `Drowsiness_Detection.ipynb` in Jupyter Notebook to train or retrain the model on the dataset.
2. **Live Detection**: Run `live_webcam_detection.py` to initiate real-time drowsiness detection from your webcam. The system will alert the user if a drowsy state is detected.

## Dataset

The dataset used to train the model can be accessed from Kaggle: [MRL Eye Dataset](https://www.kaggle.com/datasets/prasadvpatil/mrl-dataset).

## Contributing

Contributions are welcome! If you have suggestions or improvements, please submit a pull request or open an issue.

---

This repository offers an effective tool for drowsiness detection, with potential applications in the automobile industry to improve driver safety and prevent accidents.

