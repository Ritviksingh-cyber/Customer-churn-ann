# Customer Churn Prediction using Artificial Neural Network (ANN)

## 📌 Project Overview

This project predicts whether a **bank customer is likely to churn** (leave the bank) using an **Artificial Neural Network (ANN)** built with **TensorFlow/Keras**.
The application includes a **Streamlit web interface** that allows users to input customer information and instantly receive a churn prediction probability.

The project demonstrates an **end-to-end machine learning workflow** including:

* Data preprocessing
* Feature encoding
* Feature scaling
* ANN model training
* Model evaluation
* Model deployment using Streamlit

---

## 🧠 Machine Learning Model

The model is a **feed-forward Artificial Neural Network (ANN)** trained on the **Bank Customer Churn dataset**.

### Model Pipeline

```
Raw Customer Data
       ↓
Label Encoding (Gender)
       ↓
One-Hot Encoding (Geography)
       ↓
Feature Scaling (StandardScaler)
       ↓
Artificial Neural Network
       ↓
Churn Probability Prediction
```

### ANN Architecture

```
Input Layer (12 features)
      ↓
Dense Layer (64 neurons, ReLU)
      ↓
Dense Layer (32 neurons, ReLU)
      ↓
Dense Layer (16 neurons, ReLU)
      ↓
Output Layer (Sigmoid)
```

### Loss Function

```
Binary Crossentropy
```

### Optimizer

```
Adam Optimizer
```

---

## 📊 Dataset

Dataset used: **Bank Customer Churn Modelling Dataset**

Features include:

| Feature         | Description                    |
| --------------- | ------------------------------ |
| CreditScore     | Customer credit score          |
| Geography       | Customer country               |
| Gender          | Male/Female                    |
| Age             | Customer age                   |
| Tenure          | Years with bank                |
| Balance         | Account balance                |
| NumOfProducts   | Number of bank products        |
| HasCrCard       | Credit card ownership          |
| IsActiveMember  | Customer activity status       |
| EstimatedSalary | Estimated salary               |
| Exited          | Target variable (Churn or Not) |

---

## 🛠 Tech Stack

**Programming Language**

* Python

**Machine Learning**

* TensorFlow
* Keras
* Scikit-Learn

**Data Processing**

* Pandas
* NumPy

**Visualization & Monitoring**

* TensorBoard

**Deployment**

* Streamlit

---

## 📂 Project Structure

```
ANN
│
├── app.py                         # Streamlit web application
├── ann_model.h5                   # Trained ANN model
├── model.h5                       # Saved model file
├── scaler.pkl                     # StandardScaler object
├── geo.pkl                        # Geography encoder
├── label_encoder_gender.pkl       # Gender encoder
├── onehot_encoder_geo.pkl         # OneHot encoder for geography
├── Churn_Modelling.csv            # Dataset
├── Experiment.ipynb               # Training notebook
├── prediction.ipynb               # Prediction pipeline notebook
├── requirements.txt               # Python dependencies
└── logs/                          # TensorBoard logs
```

---

## 🚀 Running the Project

### 1️⃣ Clone the Repository

```
git clone https://github.com/yourusername/customer-churn-ann.git
cd customer-churn-ann
```

### 2️⃣ Create Virtual Environment

```
python -m venv .venv
source .venv/bin/activate
```

### 3️⃣ Install Dependencies

```
pip install -r requirements.txt
```

### 4️⃣ Run Streamlit App

```
streamlit run app.py
```

The app will open at:

```
http://localhost:8501
```

---

## 🖥 Streamlit Web Application

The Streamlit interface allows users to:

* Input customer details
* Run ANN prediction
* View churn probability
* Receive churn risk classification

Example output:

```
Churn Probability: 0.27
Customer is not likely to churn
```

---

## 📈 Model Monitoring

TensorBoard is used to visualize:

* Training loss
* Validation loss
* Accuracy curves
* Model graph

Run TensorBoard:

```
tensorboard --logdir logs/fit
```

Open:

```
http://localhost:6006
```

---

## 📌 Key Learnings

This project demonstrates:

* Feature preprocessing for tabular data
* Neural network training using TensorFlow
* Model serialization with pickle
* Building ML inference pipelines
* Deploying ML models with Streamlit

---

## 👨‍💻 Author

**Ritvik Singh**

AI/ML Enthusiast | Software Developer
Focused on building practical machine learning systems and AI applications.

---

## ⭐ If you found this project useful

Consider **starring the repository** to support the work.
