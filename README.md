# Tuberculosis Detection Using Machine Learning - Samiksha Rajesh Dixit NUID: 002831689 DS 5220 
This repository contains the implementation of a supervised machine learning project focused on predicting tuberculosis (TB) based on patient data. The project evaluates multiple classification algorithms to identify the most effective model for TB diagnosis.

## Introduction
Tuberculosis remains a significant public health issue, particularly in low-resource settings. This project aims to leverage machine learning to create a scalable, efficient diagnostic model. By using a structured dataset of patient information, we evaluate the performance of various supervised learning algorithms, including Logistic Regression, Random Forest, Decision Trees, and Support Vector Machines.

The best-performing model can serve as a foundation for developing real-world diagnostic tools.

The repository contains the following files:

- Tuberculosis_code.ipynb: The Jupyter notebook containing the code for data preprocessing, model training, evaluation, and results visualization.
- data/: Placeholder for the dataset (not included for privacy reasons).
- README.md: Documentation for the project.

## Installation
To run this project, ensure you have Python 3.x installed. Follow the steps below:

## Clone the repository:

bash
Copy code
git clone https://github.com/samikshardixit/SML-Tuberculosis-detection.git
cd tuberculosis-detection
Create a virtual environment (optional but recommended):

bash
Copy code
python -m venv env
source env/bin/activate  # On Windows, use `env\Scripts\activate`
Install the required dependencies:

bash
Copy code
pip install -r requirements.txt
(Ensure requirements.txt lists all necessary libraries, such as scikit-learn, pandas, numpy, matplotlib, etc.)

## Usage
Load the Jupyter Notebook
Open the Tuberculosis_code.ipynb file in Jupyter Notebook or any compatible environment (e.g., Google Colab).

## Prepare the Dataset

Download your dataset and place it in the data/ directory.
Ensure the dataset matches the expected format (e.g., CSV file with specified feature names).
Run the Notebook
Execute the notebook cells step-by-step:

Data Preprocessing: Handles missing values, normalizes features, and splits the dataset.
Model Training: Trains models (Logistic Regression, Random Forest, etc.) on the training set.
Evaluation: Generates performance metrics like accuracy, precision, recall, and confusion matrices.
Visualization: Plots ROC and Precision-Recall curves for model comparison.
Parameter Tuning (Optional)
Customize the hyperparameters in the model training section to experiment with different configurations.

## Results
The best-performing model was Random Forest, which achieved:

F1 - Score: 0.67
Precision: 0.95
Recall: 0.56
Other models, such as Logistic Regression and SVM, were tested but did not perform as well. 

We welcome contributions to enhance this project! To contribute:

Fork the repository.
Create a feature branch:
bash
Copy code
git checkout -b feature-name
Commit your changes:
bash
Copy code
git commit -m "Add feature-name"
Push to the branch:
bash
Copy code
git push origin feature-name
Open a pull request.
References
Dataset: [Dataset 1] (https://www.kaggle.com/datasets/iamtapendu/chest-x-ray-lungs-segmentation?select=MetaData.csv) [Dataset 2] (https://www.kaggle.com/datasets/beosup/lung-segment?select=masks) [Dataset 3] (https://www.kaggle.com/datasets/nih-chest-xrays/data)
Libraries: scikit-learn, pandas, matplotlib

