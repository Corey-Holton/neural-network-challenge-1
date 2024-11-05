# Student Loan Risk Prediction with Deep Learning

This project aims to predict the risk of student loan repayment using a deep learning model built with TensorFlow. By analyzing a dataset of student loan information, the model can help assess the risk level associated with each loan, potentially informing lending decisions.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Data Preparation](#data-preparation)
6. [Model Architecture](#model-architecture)
7. [Model Training](#model-training)
8. [Evaluation](#evaluation)
9. [Saving and Reloading the Model](#saving-and-reloading-the-model)
10. [Generating Predictions](#generating-predictions)
11. [Future Improvements](#future-improvements)
12. [License](#license)

## Project Overview
The objective of this project is to build and evaluate a neural network that predicts the likelihood of student loan repayment. By leveraging a binary classification approach, the model classifies loans based on a `credit_ranking` variable, which serves as the target for this prediction task.

## Dataset
The project uses the `student-loans.csv` dataset, which contains features relevant to student loan characteristics, including demographic and credit-related information. The target variable, `credit_ranking`, categorizes the credit status of each loan.

### Features
- Various features from the dataset are used to define the input data (`X`).
- The `credit_ranking` column serves as the target variable (`y`).

### Target
The `credit_ranking` variable represents loan risk levels, used here as the classification target.

### Installation
To run this project, ensure you have the following libraries installed:

- pandas
- tensorflow
- scikit-learn
## Installation via pip
pip install pandas tensorflow scikit-learn

## Data Preparation
The data preparation steps include loading, reviewing, and preprocessing the data to get it ready for training. Key steps include:

1. Loading the data from a CSV file.
2. Defining the target (y) and features (X) datasets.
3. Splitting the data into training and testing sets.
4. Scaling the features using StandardScaler from scikit-learn to standardize the data.

## Model Architecture
The neural network model is built with TensorFlow's Keras library and has the following structure:

- **Input Layer**: Matches the number of features.
- **Hidden Layers**:

    -  Layer 1: 6 neurons, ReLU activation.
    -  Layer 2: 3 neurons, ReLU activation.
- **Output Layer**: 1 neuron, sigmoid activation (for binary classification).

## Model Summary
The model architecture can be viewed using nn_model.summary() after it is compiled.

Model Training
The model is compiled and trained using the following configuration:

Loss Function: binary_crossentropy
Optimizer: adam
Metrics: accuracy
The model is trained over 50 epochs using the training dataset.

python
Copy code
# Compile the Sequential model
nn_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Fit the model using the training data
fit_model = nn_model.fit(X_train_scaled, y_train, epochs=50)
Evaluation
The model is evaluated on the testing dataset to assess its performance based on:

Loss
Accuracy
These metrics are printed to the console to provide insights into the model's ability to classify loan risks effectively.

python
Copy code
# Evaluate the model using test data
model_loss, model_accuracy = nn_model.evaluate(X_test_scaled, y_test, verbose=2)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")
Saving and Reloading the Model
The trained model is saved as a .keras file for future use. This allows the model to be reloaded and used for predictions without retraining.

python
Copy code
# Save the model to a keras file
file_path = Path("student_loans.keras")
nn_model.save(file_path)

# Reload the saved model
reloaded_model = tf.keras.models.load_model(file_path)
Generating Predictions
The model can generate predictions on new or testing data. The predictions are rounded to binary outcomes for classification purposes, and the results are saved to a DataFrame for analysis.

python
Copy code
# Generate predictions with the test data
predictions = reloaded_model.predict(X_test_scaled, verbose=2)
predictions_rounded = [round(prediction[0], 0) for prediction in predictions]
predictions_df = pd.DataFrame(predictions_rounded, columns=['Predicted'])
Future Improvements
Future enhancements could include:

Data Augmentation: Collecting more diverse data to improve generalization.
Hyperparameter Tuning: Experimenting with different model architectures and hyperparameters.
Feature Engineering: Creating additional features to better capture loan characteristics.
Recommendation System: Developing a recommendation engine to suggest suitable loans for students based on their risk profile.
License
This project is licensed under the MIT License. See the LICENSE file for details.

# Conclusion
To build a recommendation system for student loans, we need to collect various types of data, including demographic information (like age and marital status), educational background (such as GPA and field of study), financial details (like income and credit history), loan history (including past loans and repayment patterns), and the student’s preferences for loan amounts and terms. This data helps tailor loan options to each student’s financial situation and repayment capability. The system should use content-based filtering, which matches students to loans based on their specific profiles, making it effective even without a large user base. However, challenges include ensuring data privacy and security due to the sensitivity of financial information, maintaining data accuracy to provide reliable recommendations, ensuring the model is easy to understand for users, complying with fair lending regulations, and accounting for the uncertainties in a student’s future financial situation that could affect loan repayment.