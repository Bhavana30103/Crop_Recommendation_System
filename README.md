# 🌾 Crop Recommendation System using Machine Learning & Flask

A smart web-based **Crop Recommendation System** that suggests the most suitable crop for cultivation based on soil nutrients and environmental conditions.  
This project combines **Machine Learning** and a **Flask web application** to make agriculture more data-driven and efficient.

---

## 📖 Overview

Farming decisions are often based on guesswork, which can lead to reduced yield.  
This system helps farmers and researchers by predicting the **ideal crop** to cultivate, using parameters such as:

- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature 🌡️
- Humidity 💧
- pH
- Rainfall ☔

---

## 🧠 Machine Learning Workflow

### **1. Dataset**
- Public dataset containing soil & environmental features
- Target: **Crop label**

### **2. Preprocessing**
- Handled missing values  
- Normalized features  
- Converted data into ML-ready format  

### **3. Model Training**
- Algorithms explored: Decision Tree, Random Forest, Naïve Bayes  
- Best-performing model selected and trained  
- Accuracy evaluated using test set  

### **4. Save Model**
- Final trained model saved using **Pickle**:
```python
import pickle
pickle.dump(model, open("crop_recommendation_best.pkl", "wb"))

