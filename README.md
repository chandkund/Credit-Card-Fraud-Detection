# Credit Card Fraud Detection using Machine Learning 💳🔍

## Overview
This project aims to detect fraudulent credit card transactions using various machine learning models. Given the sensitive nature of financial transactions, the goal is to develop a reliable model that can distinguish between fraudulent and non-fraudulent transactions with high accuracy.

## Table of Contents
- [Project Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Data Exploration](#data-exploration)
- [Data Preprocessing](#data-preprocessing)
- [Model Building](#model-building)
- [Model Evaluation](#model-evaluation)
- [Results](#results)
- [Conclusion](#conclusion)
- [Future Work](#future-work)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Dataset
The dataset used in this project contains credit card transactions made by European cardholders in September 2013. It consists of 31 features, including 28 anonymized variables (due to confidentiality), transaction amount, and a class label indicating whether the transaction is fraudulent or not.

- **Total transactions:** 284,807
- **Fraudulent transactions:** 492

The dataset is highly imbalanced, with fraudulent transactions representing only 0.17% of all transactions.

## Installation
To run this project, clone the repository and install the required packages.

```bash
git clone https://github.com/chandkumar/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
pip install -r requirements.txt
```

## Data Exploration
An initial exploratory data analysis (EDA) was conducted to understand the distribution and relationships within the data. Key insights include:
- Visualization of the distribution of fraudulent vs. non-fraudulent transactions.
- Analysis of anonymized features to identify patterns indicative of fraud.

## Data Preprocessing
Data preprocessing involved the following steps:
- **Handling Imbalanced Data:** Techniques like undersampling, oversampling, and SMOTE were employed to address class imbalance.
- **Outlier Detection:** Identified and removed outliers based on the interquartile range (IQR).
- **Feature Scaling:** Normalized transaction amounts to ensure consistent model performance.
- **Train-Test Split:** The dataset was split into training and testing sets for model evaluation.

## Model Building
Four machine learning models were implemented:
1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **Support Vector Classifier (SVC)**
4. **Random Forest Classifier**

Each model was fine-tuned and evaluated for its ability to detect fraudulent transactions.

## Model Evaluation
Models were evaluated using the following metrics:
- **Accuracy**

The evaluation focused on minimizing false negatives, as missing fraudulent transactions is more critical than misclassifying non-fraudulent ones.

## Results
Logistic Regression achieved the highest performance, with an accuracy of 93.7%  This model successfully identified a significant portion of fraudulent transactions while maintaining a low false positive rate.

## Conclusion
This project demonstrates the application of machine learning models in detecting credit card fraud. By combining different techniques and models, a robust system was developed to enhance financial security.

## Future Work
- **Feature Engineering:** Explore additional features to improve model accuracy.
- **Ensemble Methods:** Combine multiple models to create a more powerful fraud detection system.
- **Real-time Detection:** Implement the model in a real-time environment for immediate fraud detection.

## Acknowledgements
- The dataset was provided by [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud).
- Thanks to the scikit-learn community for the comprehensive documentation and resources.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
