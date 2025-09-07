# 🌾 Crop Recommendation System using Machine Learning & Flask

This project is a **Crop Recommendation System** that uses a **Machine Learning model** and a **Flask web application** to suggest the most suitable crop for cultivation based on soil nutrients and environmental conditions.

---

## 📖 Project Workflow 

### **1. Problem Understanding**
- Farmers often struggle to decide the best crop to grow based on soil fertility and weather conditions.
- This system helps by predicting the ideal crop using data-driven insights.

---

### **2. Data Collection & Preprocessing**
- Dataset containing soil nutrients (N, P, K), weather details (temperature, humidity, rainfall), and pH levels was used.  
- Preprocessing steps:
  - Handling missing values
  - Normalizing data
  - Feature engineering 

---

### **3. Model Training**
- A Machine Learning model (trained in Python using **scikit-learn**) was created.  
- The trained model was saved using **Pickle** as `crop_recommendation_best.pkl`.  

```python
import pickle
# Example: saving model after training
pickle.dump(model, open("crop_recommendation_best.pkl", "wb"))

