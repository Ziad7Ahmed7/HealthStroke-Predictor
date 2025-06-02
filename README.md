# HealthStroke-Predictor

## Stroke Prediction Model

This project implements a machine learning model to predict the likelihood of stroke based on various health and lifestyle parameters. The model utilizes a Decision Tree Classifier and addresses class imbalance using SMOTE. A user-friendly graphical interface (GUI) is developed using customtkinter to allow for easy interaction and prediction.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Tools and Libraries](#tools-and-libraries)
- [Installation](#installation)
- [Usage](#usage)
- [Data Preprocessing Steps](#data-preprocessing-steps)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

Stroke is a serious medical condition that occurs when the blood supply to part of the brain is interrupted or reduced, depriving brain tissue of oxygen and nutrients. This project aims to build a predictive model that can assist in identifying individuals at higher risk of stroke, enabling early intervention and preventive measures.

The project covers the entire machine learning pipeline from data loading and preprocessing to model training, evaluation, and deployment via a simple GUI.

## Features

- **Data Loading and Exploration**: Reads and displays initial insights from the stroke dataset.
- **Class Imbalance Handling**: Employs SMOTE to address the skewed distribution of the target variable ('stroke').
- **Data Preprocessing**: Handles missing values, performs feature selection, and encodes categorical variables.
- **Decision Tree Classifier**: Trains a Decision Tree model for stroke prediction.
- **Model Evaluation**: Assesses model performance using various metrics (accuracy, F1-score, precision, recall, ROC AUC).
- **Hyperparameter Tuning**: Uses GridSearchCV to find optimal model parameters.
- **Interactive GUI**: A custom Tkinter-based interface for making predictions.

## Dataset

The dataset used in this project is `healthcare-dataset-stroke-data.csv`. It contains information about patients, including:

- **id**: Unique identifier
- **gender**: Gender of the patient
- **age**: Age of the patient
- **hypertension**: 0 if the patient has no hypertension, 1 if the patient has hypertension
- **heart_disease**: 0 if the patient has no heart disease, 1 if the patient has heart disease
- **ever_married**: "Yes" or "No"
- **work_type**: "children", "Govt_job", "Never_worked", "Private", "Self-employed"
- **Residence_type**: "Rural" or "Urban"
- **avg_glucose_level**: Average glucose level in blood
- **bmi**: Body Mass Index
- **smoking_status**: "formerly smoked", "never smoked", "smokes", "Unknown"
- **stroke**: 1 if the patient had a stroke, 0 if not (Target Variable)

## Tools and Libraries

The project is developed using Python and the following key libraries:

- **pandas**: Data manipulation and analysis.
- **numpy**: Numerical operations.
- **matplotlib**: Data visualization.
- **seaborn**: Enhanced data visualization.
- **scikit-learn (sklearn)**: Machine learning algorithms (Decision Tree Classifier, model selection, metrics).
- **imbalanced-learn (imblearn)**: For handling imbalanced datasets (SMOTE).
- **tkinter**: Standard Python GUI toolkit.
- **customtkinter**: Modern and customizable Tkinter widgets for a better UI.
- **joblib**: For saving and loading Python objects (trained models).
- **Pillow (PIL)**: Image processing capabilities for GUI elements.

## Installation

To run this project, you need to have Python installed. It's recommended to use a virtual environment.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/stroke-prediction.git
   cd stroke-prediction
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   # On Windows
   .\venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install the required libraries:**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn customtkinter pillow
   ```

## Usage

1. **Download the dataset:**
   Ensure the `healthcare-dataset-stroke-data.csv` file is in the root directory of your project.

2. **Run the Jupyter Notebook:**
   Open the `DescionTreeClassfier_Strokes.ipynb` notebook in a Jupyter environment (e.g., Jupyter Lab, Jupyter Notebook, VS Code with Python extension) and run all cells. This will train the model and, if implemented, launch the GUI.

   ```bash
   jupyter notebook
   ```

   Then, navigate to and open `DescionTreeClassfier_Strokes.ipynb`.

3. **Use the GUI (if implemented in the notebook):**
   Once the notebook runs completely, a GUI window should appear, allowing you to input patient details and get a stroke prediction.

## Data Preprocessing Steps

The notebook performs the following preprocessing steps:

1. **Loading Data**: Reads `healthcare-dataset-stroke-data.csv`.

2. **Handling NaN Values**: Removes rows with missing bmi values using `df.dropna()`.

3. **Feature Selection**: Drops `id`, `work_type`, `Residence_type`, and `smoking_status` columns.

4. **Categorical Encoding**:
   - **gender**: 'Male' -> 1, 'Female' -> 0.
   - **ever_married**: 'Yes' -> 1, 'No' -> 0.
   - **smoking_status**: 'formerly smoked' -> 1, 'smokes' -> 1, 'never smoked' -> 0 (Note: "Unknown" values would need to be handled, either by dropping or imputation, which is not fully visible in the provided snippet but implied by the smoking_mapping definition).

5. **Class Imbalance**: Addresses the imbalance in the 'stroke' target variable using SMOTE to oversample the minority class.

## Model Training and Evaluation

- **Model**: Decision Tree Classifier (`sklearn.tree.DecisionTreeClassifier`).
- **Splitting Data**: The dataset is split into training and testing sets.
- **Hyperparameter Tuning**: GridSearchCV is used to optimize the Decision Tree's hyperparameters for best performance.
- **Metrics**: The model's performance is evaluated using:
  - Accuracy Score
  - Confusion Matrix
  - F1-score
  - Recall
  - Precision
  - ROC Curve and AUC (Area Under the Curve)
  - Classification Report

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.

## License

This project is open-source and available under the MIT License.
