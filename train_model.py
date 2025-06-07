import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Generate mock data for predictive maintenance
def generate_mock_data(num_samples=1000):
    np.random.seed(42)
    data = {
        "Hours_Used": np.random.randint(1, 1000, num_samples),
        "Temperature": np.random.normal(70, 15, num_samples).round(1),
        "Vibration_Level": np.random.normal(5, 2, num_samples).round(2),
        "Days_Since_Last_Service": np.random.randint(1, 365, num_samples),
    }
    
    # Simulate failure conditions (higher hours + high temp/vibration)
    failure_conditions = (
        (data["Hours_Used"] > 800) |
        (data["Temperature"] > 85) |
        (data["Vibration_Level"] > 8) |
        (data["Days_Since_Last_Service"] > 300)
    )
    data["Failed"] = np.where(failure_conditions, 1, 0)  # 1 = Failure, 0 = Healthy
    
    return pd.DataFrame(data)

# Generate and save mock data
df = generate_mock_data()
df.to_csv("predictive_maintenance_data.csv", index=False)

# Train a Random Forest model
X = df.drop("Failed", axis=1)
y = df["Failed"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(model, "predictive_maintenance_model.joblib")
print("Model saved!")