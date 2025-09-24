# Housing Price Prediction Model

## 📌 Overview
This project is my **first machine learning model** (apart from the Iris dataset), where I explored real-world data processing and predictive modeling.  
The goal was to predict housing prices based on various features such as location, income levels, and housing attributes.  

It demonstrates how raw data can be transformed into meaningful insights through **data preprocessing**, **model training**, and **inference pipelines**.  

---

## 🏗 Project Structure
The project contains a dataset of housing information and a Python script that handles both training and prediction.  
It uses a preprocessing pipeline to handle missing values, scale numerical features, and encode categorical ones.  
A **RandomForestRegressor** is trained on the prepared data to predict the median house value.

---

## 🧠 Key Learnings
Through this project, I learned:
- How to handle missing data and categorical features using preprocessing pipelines.
- The importance of stratified sampling to create balanced training and testing datasets.
- How to serialize models and preprocessing pipelines for later use.
- How to deploy a trained model for inference without retraining.

This project marked a significant step in my machine learning journey after the Iris dataset, showing how to move from raw data to a working predictive model.

---

## 🎯 Purpose
The aim of this project was not just to build a working model, but to **understand the complete ML workflow**:
- Data preparation
- Pipeline creation
- Model training
- Saving and loading models for inference  
This serves as a foundation for more advanced projects involving larger datasets and complex models.

---

## 💡 Future Improvements
Future versions of this project could include:
- Hyperparameter tuning for better model performance
- Cross-validation for more robust evaluation
- Deployment of the model via a Flask API for real-time predictions


---

### 💡 Tip
This project was designed as a learning exercise to understand end-to-end machine learning workflows.  
For better model performance in the future, experimenting with different regression models and hyperparameter tuning is recommended.

---

## 📜 License
This project is licensed under the MIT License.
