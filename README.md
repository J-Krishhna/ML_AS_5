# Breast Cancer Diagnosis using K-Nearest Neighbors

This project predicts whether a tumor is benign or malignant using selected medical features and the K-Nearest Neighbors (KNN) classification algorithm.

## Dataset
- **Source:** Kaggle – Breast Cancer Dataset  
- https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset

## Objectives
- Import and inspect the dataset using Pandas  
- Select relevant numerical features for classification  
- Encode the target variable (Diagnosis)  
- Apply feature scaling for distance-based learning  
- Train a KNN classifier  
- Experiment with different values of K  
- Evaluate model performance using accuracy, precision, recall, and F1-score  
- Visualize results using confusion matrix and decision boundary  

## Key Insights
- Feature scaling significantly improves KNN performance  
- The optimal K value balanced bias and variance (best K = 11)  
- The model achieved approximately 93% accuracy  
- Small K values caused overfitting, while large K values led to underfitting  
- KNN is sensitive to irrelevant features and high-dimensional data  

# Loan Approval Prediction using Decision Tree

This project predicts whether a loan application will be approved or rejected using applicant and loan-related features with a Decision Tree classifier.

## Dataset
- **Source:** Kaggle – Loan Prediction Dataset  
- https://www.kaggle.com/datasets/ninzaami/loan-predication

## Objectives
- Load and inspect the loan dataset  
- Handle missing values appropriately  
- Encode categorical variables  
- Split the dataset into training and testing sets  
- Train a Decision Tree classifier  
- Experiment with tree depth and pruning  
- Evaluate performance using accuracy, precision, recall, and F1-score  
- Analyze feature importance  
- Compare shallow and deep trees  


## Key Insights
- Credit History was the most influential feature for loan approval  
- Deep trees achieved perfect training accuracy but lower test accuracy  
- Overfitting was observed when training accuracy reached 100%  
- A shallow tree improved generalization performance  
- Decision Trees are highly interpretable and require minimal preprocessing  
