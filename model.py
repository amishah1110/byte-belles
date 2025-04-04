import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load datasets
sleep_df = pd.read_excel("Sleep Dataset.xlsm.xlsx")
social_df = pd.read_excel("Social Media Usage - Train.xlsm.xlsx")

# Step 1: Define Age Groups (Only 18-59)
def categorize_age(age):
    if 18 <= age < 30:
        return "Young Adult (18-29)"
    elif 30 <= age < 45:
        return "Middle-aged (30-44)"
    elif 45 <= age < 60:
        return "Older Adult (45-59)"
    else:
        return "Exclude"

# Apply age categorization
sleep_df['Age Group'] = sleep_df['Age'].apply(categorize_age)
social_df['Age Group'] = social_df['Age'].apply(categorize_age)

# Remove excluded age groups
sleep_df = sleep_df[sleep_df['Age Group'] != "Exclude"]
social_df = social_df[social_df['Age Group'] != "Exclude"]

# Convert Gender to Numeric for Consistency
sleep_df['Gender'] = sleep_df['Gender'].map({'Male': 0, 'Female': 1})
social_df['Gender'] = social_df['Gender'].map({'Male': 0, 'Female': 1, 'Non-binary': 2})

# Compute Total Social Media Usage
social_df["Total Social Media Hours"] = social_df.iloc[:, 1:].sum(axis=1)

# Identify Most Used Platform
social_df["Most Used Platform"] = social_df.iloc[:, 1:-1].idxmax(axis=1)

# Merge both datasets based on Age Group & Gender
merged_df = pd.merge(sleep_df, social_df, on=["Age Group", "Gender"], how="inner")

# Encode 'Most Used Platform' as Numeric
encoder = LabelEncoder()
merged_df["Most Used Platform"] = encoder.fit_transform(merged_df["Most Used Platform"])

# Select Relevant Features
features = ["Total Social Media Hours", "Most Used Platform", "Sleep Duration", "Stress Level", "Physical Activity Level"]

# Step 2: Train Model to Predict High Stress Level
X = merged_df[features]
y_stress = (merged_df["Stress Level"] > merged_df["Stress Level"].median()).astype(int)  # Binary classification

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y_stress, test_size=0.2, random_state=42)

# Train Random Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict & Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Step 3: Predict Poor Sleep Quality
y_sleep = (merged_df["Sleep Duration"] < merged_df["Sleep Duration"].median()).astype(int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_sleep, test_size=0.2, random_state=42)

# Train Model
sleep_model = RandomForestClassifier(n_estimators=100, random_state=42)
sleep_model.fit(X_train, y_train)

# Predict & Evaluate
y_pred_sleep = sleep_model.predict(X_test)
print("Accuracy (Sleep Quality Prediction):", accuracy_score(y_test, y_pred_sleep))
print(classification_report(y_test, y_pred_sleep))
