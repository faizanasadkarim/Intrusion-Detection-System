# Intrusion Detection System (IDS) for Network Anomaly and Attack Detection

## 🚀 Objective

This project aims to address network security challenges by detecting and classifying anomalies and malicious activities in network traffic. The **Intrusion Detection System (IDS)** identifies deviations from normal network behavior and accurately classifies potential attacks.

### 📊 Key Features:
- **Preprocessing network traffic data** and engineering features to optimize classification accuracy.
- Achieved **99.7% accuracy** for anomaly detection using **Logistic Regression**.
- Achieved **99.99% accuracy** for attack classification using **Random Forest**.

---

## 🛠 Tools & Libraries

- **Pandas**: Data manipulation and analysis.
- **NumPy**: Efficient numerical computation.
- **Scikit-learn**: Machine learning algorithms and tools for model building and evaluation.
- **Matplotlib**: Data visualization for understanding patterns in network traffic.
- **Seaborn**: Statistical data visualization for better insights.

---

## 🔧 Techniques Used

- **Feature Engineering**: Extracting meaningful features from raw network data to improve model performance.
- **Label Encoding**: Converting categorical labels into numeric form for model training.
- **Classification**: Building models to classify network traffic into normal or attack categories.
- **Precision/Recall Evaluation**: Evaluating model performance using precision and recall metrics to ensure accuracy and minimize false positives.

---

## 📈 Results

- **Anomaly Detection Accuracy**: **99.7%** using Logistic Regression.
- **Attack Classification Accuracy**: **99.99%** using Random Forest.

---

## 📥 Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/faizanasadkarim/Intrusion-Detection-System.git




## ⚡ Usage
Load the preprocessed network traffic dataset.
Perform feature engineering and label encoding.
Train the model using Logistic Regression for anomaly detection and Random Forest for attack classification.
Evaluate the model using precision, recall, and accuracy metrics.
Visualize results using Matplotlib and Seaborn.

## 📝 Example
```
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
# Load data
data = pd.read_csv("network_traffic_data.csv")

# Feature engineering and model training
X = data.drop('label', axis=1)
y = data['label']

# Train models
logistic_model = LogisticRegression()
logistic_model.fit(X, y)

rf_model = RandomForestClassifier()
rf_model.fit(X, y)

# Evaluate the model
print(classification_report(y, logistic_model.predict(X)))
print(classification_report(y, rf_model.predict(X)))
```
## 🤝 Contributing
Feel free to fork the repository, raise issues, and submit pull requests. Contributions to enhance the model’s accuracy or add new features are welcome.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## ✨ Acknowledgements
Special thanks to the contributors and the open-source community for their valuable resources.

