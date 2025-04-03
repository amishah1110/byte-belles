import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

# Define dataset folder
dataset_folder = "C:\\Users\\ishit\\PycharmProjects\\datathon 2025\\data"

# Define file paths dynamically
file_paths = {
    "train": os.path.join(dataset_folder, "Social Media Usage - Train.xlsm"),
    "test": os.path.join(dataset_folder, "Social Media Usage - Test.xlsm"),
    "val": os.path.join(dataset_folder, "Social Media Usage - Val.xlsm"),
    "sleep": os.path.join(dataset_folder, "Sleep Dataset.xlsm")
}

# Load datasets
train_df = pd.read_excel(file_paths["train"])
test_df = pd.read_excel(file_paths["test"])
val_df = pd.read_excel(file_paths["val"])
sleep_df = pd.read_excel(file_paths["sleep"], sheet_name="Sleep Dataset")

# Remove completely blank rows from all datasets
for df in [train_df, test_df, val_df, sleep_df]:
    df.dropna(how='all', inplace=True)
    df.reset_index(drop=True, inplace=True)

# Copy train_df to preserve original data
processed_df = train_df.copy()

# Identify numerical and categorical columns
num_cols = ["Daily_Usage_Time (minutes)", "Posts_Per_Day", "Likes_Received_Per_Day",
            "Comments_Received_Per_Day", "Messages_Sent_Per_Day"]

cat_cols = ["User_ID", "Age", "Gender", "Platform", "Dominant_Emotion"]

print("\nNumerical Columns:", num_cols)
print("Categorical Columns:", cat_cols)

# Convert numerical columns to float in processed_df
for col in num_cols:
    if col in processed_df.columns:
        processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')

# Apply preprocessing ONLY to training dataset
if not processed_df.empty:
    processed_df[num_cols] = processed_df[num_cols].apply(lambda x: x.fillna(x.median()), axis=0)
    for col in cat_cols:
        processed_df[col] = processed_df[col].astype(str)  # Convert to string
        mode_value = processed_df[col].mode().astype(str)[0]  # Get mode
        processed_df[col] = processed_df[col].fillna(mode_value)  # Assign properly

# Print missing values after cleaning
print("\nMissing Values After Cleaning (Train Only):")
print(processed_df.isnull().sum())

print("\n✅ Data Preprocessing Completed Successfully (Only for Training Data)")

from sklearn.preprocessing import OrdinalEncoder

# Encode categorical features using OrdinalEncoder
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# Ensure categorical columns are strings and fill NaNs
for col in ["Gender", "Platform", "Dominant_Emotion"]:
    if col in processed_df.columns:
        processed_df[col] = processed_df[col].astype(str).fillna("Unknown")
        processed_df[col] = encoder.fit_transform(processed_df[[col]])

print(train_df.head())


def remove_outliers(df, columns):
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

processed_df = remove_outliers(processed_df, num_cols)
print("\nOutliers Removed from Numerical Features.")

# Engagement Score (weighted sum of activities)
processed_df["Engagement_Score"] = (
    (processed_df["Daily_Usage_Time (minutes)"] * 0.4) +
    (processed_df["Posts_Per_Day"] * 0.2) +
    (processed_df["Likes_Received_Per_Day"] * 0.15) +
    (processed_df["Comments_Received_Per_Day"] * 0.15) +
    (processed_df["Messages_Sent_Per_Day"] * 0.1)
)

# Convert Age to numeric and create Age Groups
processed_df["Age"] = pd.to_numeric(processed_df["Age"], errors='coerce')
def age_grouping(age):
    if pd.isna(age): return "Unknown"
    age = int(age)
    return "Under 18" if age < 18 else "18-24" if age < 25 else "25-34" if age < 35 else "35-44" if age < 45 else "45-59" if age < 60 else "60+"
processed_df["Age_Group"] = processed_df["Age"].apply(age_grouping)

# Encode Age_Group
processed_df[["Age_Group"]] = encoder.fit_transform(processed_df[["Age_Group"]])
print("\nFeature Engineering Completed.")

### ENCODE CATEGORICAL VARIABLES ###
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# Ensure categorical columns are strings and fill NaNs
for col in ["Gender", "Platform", "Dominant_Emotion", "Age_Group"]:
    if col in processed_df.columns:
        processed_df[col] = processed_df[col].astype(str).fillna("Unknown")

# Encode categorical columns
processed_df[["Gender", "Platform", "Dominant_Emotion", "Age_Group"]] = encoder.fit_transform(
    processed_df[["Gender", "Platform", "Dominant_Emotion", "Age_Group"]]
)

print("\nCategorical Encoding Completed.")

### SCALE NUMERICAL FEATURES ###
scaler = StandardScaler()
scaled_features = num_cols + ["Engagement_Score"]
processed_df[scaled_features] = scaler.fit_transform(processed_df[scaled_features])

print("\nFeature Scaling Completed.")

processed_data_path = os.path.join(dataset_folder, "processed_train_data.csv")
processed_df.to_csv(processed_data_path, index=False)
print(f"\nProcessed Data Saved at: {processed_data_path}")

# Print top 5 rows for comparison
print("\nOriginal Train Data:")
print(train_df.head())
print("\nProcessed Data:")
print(processed_df.head())


# Data Visualization
plt.figure(figsize=(12, 6))
for col in num_cols:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(train_df[col], bins=30, kde=True, ax=axes[0])
    axes[0].set_title(f'Unprocessed {col}')
    sns.histplot(processed_df[col], bins=30, kde=True, ax=axes[1])
    axes[1].set_title(f'Processed {col}')
    plt.show()

# Boxplots to check outliers
for col in num_cols:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.boxplot(y=train_df[col], ax=axes[0])
    axes[0].set_title(f'Unprocessed {col}')
    sns.boxplot(y=processed_df[col], ax=axes[1])
    axes[1].set_title(f'Processed {col}')
    plt.show()

# Count plots for categorical features
for col in ["Age", "Gender", "Platform", "Dominant_Emotion"]:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.countplot(x=train_df[col], ax=axes[0], order=train_df[col].value_counts().index)
    axes[0].set_title(f'Unprocessed {col}')
    sns.countplot(x=processed_df[col], ax=axes[1], order=processed_df[col].value_counts().index)
    axes[1].set_title(f'Processed {col}')
    plt.xticks(rotation=45)
    plt.show()

print("\nData Visualization Completed.")