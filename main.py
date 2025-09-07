import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

# ---------------------------
# Step 1: Load Dataset
# ---------------------------
def load_data(path="Crop_recommendation.csv"):
    df = pd.read_csv(path)
    print("✅ Dataset loaded. Shape:", df.shape)
    return df

# ---------------------------
# Step 2: Feature Engineering
# ---------------------------
def feature_engineering(df):
    df['N_P'] = df['N'] / (df['P'] + 1e-6)
    df['N_K'] = df['N'] / (df['K'] + 1e-6)
    df['temp_hum'] = df['temperature'] * df['humidity']
    df['ph_cat'] = pd.cut(df['ph'], bins=[0, 5.5, 7.5, 14],
                          labels=['acidic', 'neutral', 'alkaline'])
    print("✅ Feature engineering complete. New columns added.")
    return df

# ---------------------------
# Step 3: Train & Save Model
# ---------------------------
def train_and_save(df):
    X = df.drop('label', axis=1)
    y = df['label']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Preprocessor
    num_features = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features)
    ])

    # Model pipeline
    rf_pipeline = Pipeline([
        ('pre', preprocessor),
        ('clf', RandomForestClassifier(n_estimators=300, random_state=42))
    ])

    # Train
    rf_pipeline.fit(X_train, y_train)
    preds = rf_pipeline.predict(X_test)

    # Evaluation
    print("\n🎯 Model Performance")
    print("Accuracy:", accuracy_score(y_test, preds))
    print("\nClassification Report:\n", classification_report(y_test, preds))

    # Save model
    joblib.dump(rf_pipeline, "crop_recommendation_best.pkl")
    print("✅ Model saved as crop_recommendation_best.pkl")

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    data = load_data("Crop_recommendation.csv")
    data = feature_engineering(data)
    train_and_save(data)

