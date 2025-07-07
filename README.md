# Logistic Regression from Scratch

This project implements a **logistic regression classifier** from scratch in Python, without using machine learning libraries like scikit-learn. It covers the entire workflow of a binary classification model, including:

- Generating synthetic two-class data.
- Defining the sigmoid activation function.
- Calculating the logistic loss (binary cross-entropy).
- Computing gradients for the weights and bias.
- Optimizing parameters using gradient descent.
- Predicting classes on new data.
- Visualizing the decision boundary and data points.

## How it works

1. **Data Generation:**  
   The code creates a synthetic dataset with two Gaussian-distributed classes.

2. **Model Definition:**  
   The logistic regression model uses a sigmoid function to map linear combinations of input features to probabilities between 0 and 1.

3. **Loss Function:**  
   The loss function used is the logistic loss (binary cross-entropy), which measures how well the predicted probabilities match the true class labels.

4. **Training:**  
   Using gradient descent, the model iteratively updates weights and bias to minimize the loss over a specified number of epochs.

5. **Prediction:**  
   After training, the model predicts class labels by thresholding the sigmoid output at 0.5.

6. **Visualization:**  
   The code plots the training data along with the decision boundary, showing how the model separates the two classes.

## Usage

Run the script to train the logistic regression model on the generated dataset and see the plotted decision boundary.

```bash
python logistic_regression_scratch.py

