<h1 align="center">Diabetes Prediction Project 🩺</h1>

<p align="center">
  A machine learning project to predict the onset of diabetes based on diagnostic medical measurements. This model is trained on the PIMA Indians Diabetes Database.
</p>

<!-- Badges -->
<p align="center">
  <a href="https://github.com/your-username/diabetes-prediction/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/your-username/diabetes-prediction?style=for-the-badge" alt="License">
  </a>
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python" alt="Python Version">
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/Status-Completed-success?style=for-the-badge" alt="Project Status">
  </a>
    <a href="https://github.com/your-username/diabetes-prediction/pulls">
    <img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=for-the-badge" alt="PRs Welcome">
  </a>
</p>

---

## 📋 Table of Contents
1.  [About The Project](#about-the-project)
2.  [Dataset](#-dataset)
3.  [Key Features](#-key-features)
4.  [Technologies Used](#-technologies-used)
5.  [Getting Started](#-getting-started)
6.  [Usage](#-usage)
7.  [Model Performance](#-model-performance)
8.  [Contributing](#-contributing)
9.  [License](#-license)
10. [Contact](#-contact)

---

## 📖 About The Project

This project aims to build a machine learning model that can accurately predict whether a patient has diabetes. The prediction is based on several diagnostic attributes included in the **PIMA Indians Diabetes Database**. The primary goal is to provide an early warning system that can help individuals take preventive measures and assist healthcare professionals in making informed decisions.

The project involves a complete data science workflow:
* **Data Exploration & Cleaning:** Understanding the dataset and handling missing or erroneous values.
* **Data Visualization:** Creating plots to uncover patterns and correlations between features.
* **Feature Engineering & Selection:** Creating new features and selecting the most impactful ones for the model.
* **Model Training & Evaluation:** Building and comparing several classification models to find the best performer.
* **Deployment (Optional):** The trained model can be deployed as a web application for real-time predictions.

---

## 📊 Dataset

This project utilizes the **PIMA Indians Diabetes Database**, a well-known dataset in the machine learning community. It was originally sourced from the National Institute of Diabetes and Digestive and Kidney Diseases. All patients in this dataset are females of at least 21 years old of Pima Indian heritage.

The dataset consists of the following features:
* **Pregnancies:** Number of times pregnant
* **Glucose:** Plasma glucose concentration a 2 hours in an oral glucose tolerance test
* **BloodPressure:** Diastolic blood pressure (mm Hg)
* **SkinThickness:** Triceps skin fold thickness (mm)
* **Insulin:** 2-Hour serum insulin (mu U/ml)
* **BMI:** Body mass index (weight in kg/(height in m)^2)
* **DiabetesPedigreeFunction:** A function that scores likelihood of diabetes based on family history
* **Age:** Age in years
* **Outcome:** The target variable (0 for non-diabetic, 1 for diabetic)

---

## ✨ Key Features

* **Exploratory Data Analysis (EDA):** In-depth analysis and visualization of data to extract insights.
* **Data Preprocessing:** Robust handling of missing values (e.g., replacing '0's in `Glucose`, `BMI`, etc. with the mean/median).
* **Feature Scaling:** Standardization of features using `StandardScaler` to ensure all features contribute equally to the model's performance.
* **Model Comparison:** Training and evaluation of multiple classification algorithms, such as:
    * Logistic Regression
    * K-Nearest Neighbors (KNN)
    * Support Vector Machine (SVM)
    * Random Forest Classifier
* **Performance Metrics:** Detailed model evaluation using Accuracy, Precision, Recall, and F1-Score.

---

## 💻 Technologies Used

* **Python 3.9**
* **Libraries:**
    * **Pandas:** For data manipulation and analysis.
    * **NumPy:** For numerical operations.
    * **Matplotlib & Seaborn:** For data visualization.
    * **Scikit-learn:** For machine learning model implementation and evaluation.
* **Jupyter Notebook:** For interactive development and documentation.

---

## 🚀 Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

Make sure you have Python 3.8 or later installed on your system. You can download it from [python.org](https://www.python.org/downloads/).

### Installation

1.  **Clone the repository**
    ```sh
    git clone [https://github.com/your-username/diabetes-prediction.git](https://github.com/your-username/diabetes-prediction.git)
    ```
2.  **Navigate to the project directory**
    ```sh
    cd diabetes-prediction
    ```
3.  **Install the required packages**
    It's recommended to create a virtual environment first.
    ```sh
    # Create and activate a virtual environment (optional but recommended)
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`

    # Install dependencies
    pip install -r requirements.txt
    ```
    *(Note: You'll need to create a `requirements.txt` file by running `pip freeze > requirements.txt` in your project environment.)*

---

## 🛠️ Usage

The main logic and analysis are contained within the Jupyter Notebook (`diabetes_prediction.ipynb`).

1.  **Launch Jupyter Notebook**
    ```sh
    jupyter notebook
    ```
2.  **Open the notebook file**
    Open `diabetes_prediction.ipynb` from the Jupyter interface in your browser.
3.  **Run the cells**
    You can run the cells sequentially to see the entire process, from data loading and cleaning to model training and evaluation. The notebook is commented to explain each step.

---

## 📈 Model Performance

After training and evaluating several models, the **Random Forest Classifier** was found to provide the best balance of accuracy and generalization on the test set.

Here's a summary of the performance on the test data:

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| Logistic Regression | 77% | 0.73 | 0.58 | 0.65 |
| K-Nearest Neighbors | 75% | 0.68 | 0.61 | 0.64 |
| Support Vector Machine | 78% | 0.75 | 0.59 | 0.66 |
| **Random Forest** | **82%** | **0.79** | **0.69** | **0.74** |

The Random Forest model demonstrates strong predictive power, making it the recommended model for this task.

---

## 🤝 Contributing

Contributions make the open-source community an amazing place to learn and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request.
1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/NewFeature`)
3.  Commit your Changes (`git commit -m 'Add some NewFeature'`)
4.  Push to the Branch (`git push origin feature/NewFeature`)
5.  Open a Pull Request

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for more information.

---

## 📞 Contact

Namshima B. Iordye

[Twitter](https://x.com/Namshima001?t=M2BjOSSyH8Q6IuAQz391qw&s=09)

[Email](namshimaiordye@yahoo.com)
