import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

# Define dataset folder
dataset_folder = "C:\\Users\\amisa\\PycharmProjects\\PythonProject1\\data"

# Define file path dynamically
file_path = os.path.join(dataset_folder, "Social Media Usage - Test.xlsm")

# Load test dataset
test_df = pd.read_excel(file_path)

# Remove completely blank rows
test_df.dropna(how='all', inplace=True)
test_df.reset_index(drop=True, inplace=True)

# Define numerical and categorical columns
num_cols = ["Daily_Usage_Time (minutes)", "Posts_Per_Day", "Likes_Received_Per_Day",
            "Comments_Received_Per_Day", "Messages_Sent_Per_Day"]

cat_cols = ["User_ID", "Age", "Gender", "Platform", "Dominant_Emotion"]

# Copy test_df to preserve original data
processed_test_df = test_df.copy()

# Convert numerical columns to float
for col in num_cols:
    if col in processed_test_df.columns:
        processed_test_df[col] = pd.to_numeric(processed_test_df[col], errors='coerce')

# Handle missing values in numerical columns
if not processed_test_df.empty:
    processed_test_df[num_cols] = processed_test_df[num_cols].apply(lambda x: x.fillna(x.median()), axis=0)
    for col in cat_cols:
        processed_test_df[col] = processed_test_df[col].astype(str)
        mode_value = processed_test_df[col].mode().astype(str)[0]
        processed_test_df[col] = processed_test_df[col].fillna(mode_value)

# Remove numeric values from categorical columns
def remove_numeric_values(value):
    return value if not value.isnumeric() else None

for col in ["Gender", "Platform", "Dominant_Emotion"]:
    processed_test_df[col] = processed_test_df[col].apply(lambda x: remove_numeric_values(x)).astype(str)
    mode_value = processed_test_df[col].mode()[0] if not processed_test_df[col].mode().empty else "Unknown"
    processed_test_df[col] = processed_test_df[col].fillna(mode_value)

# Convert Age to numeric and create Age Groups
processed_test_df["Age"] = pd.to_numeric(processed_test_df["Age"], errors='coerce')

def age_grouping(age):
    if pd.isna(age): return "Unknown"
    age = int(age)
    return "Under 18" if age < 18 else "18-24" if age < 25 else "25-34" if age < 35 else "35-44" if age < 45 else "45-59" if age < 60 else "60+"

processed_test_df["Age_Group"] = processed_test_df["Age"].apply(age_grouping)

# Encode categorical variables
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
processed_test_df[["Gender", "Platform", "Dominant_Emotion", "Age_Group"]] = encoder.fit_transform(
    processed_test_df[["Gender", "Platform", "Dominant_Emotion", "Age_Group"]]
)

# Scale numerical features
scaler = StandardScaler()
scaled_features = num_cols
processed_test_df[scaled_features] = scaler.fit_transform(processed_test_df[scaled_features])

# Save processed test data
processed_test_data_path = os.path.join(dataset_folder, "processed_test_data.csv")
processed_test_df.to_csv(processed_test_data_path, index=False)

print(f"\nProcessed Test Data Saved at: {processed_test_data_path}")

# --- Validation Dataset Processing --- #

# Define validation dataset file path
val_file_path = os.path.join(dataset_folder, "Social Media Usage - Val.xlsm")

# Load validation dataset
val_df = pd.read_excel(val_file_path)

# Remove completely blank rows
val_df.dropna(how='all', inplace=True)
val_df.reset_index(drop=True, inplace=True)

# Copy val_df to preserve original data
processed_val_df = val_df.copy()

# Convert numerical columns to float
for col in num_cols:
    if col in processed_val_df.columns:
        processed_val_df[col] = pd.to_numeric(processed_val_df[col], errors='coerce')

# Handle missing values in numerical columns
if not processed_val_df.empty:
    processed_val_df[num_cols] = processed_val_df[num_cols].apply(lambda x: x.fillna(x.median()), axis=0)
    for col in cat_cols:
        processed_val_df[col] = processed_val_df[col].astype(str)
        mode_value = processed_val_df[col].mode().astype(str)[0]
        processed_val_df[col] = processed_val_df[col].fillna(mode_value)

# Remove numeric values from categorical columns
for col in ["Gender", "Platform", "Dominant_Emotion"]:
    processed_val_df[col] = processed_val_df[col].apply(lambda x: remove_numeric_values(x)).astype(str)
    mode_value = processed_val_df[col].mode()[0] if not processed_val_df[col].mode().empty else "Unknown"
    processed_val_df[col] = processed_val_df[col].fillna(mode_value)

# Convert Age to numeric and create Age Groups
processed_val_df["Age"] = pd.to_numeric(processed_val_df["Age"], errors='coerce')
processed_val_df["Age_Group"] = processed_val_df["Age"].apply(age_grouping)

# Encode categorical variables
processed_val_df[["Gender", "Platform", "Dominant_Emotion", "Age_Group"]] = encoder.transform(
    processed_val_df[["Gender", "Platform", "Dominant_Emotion", "Age_Group"]]
)

# Scale numerical features
processed_val_df[scaled_features] = scaler.transform(processed_val_df[scaled_features])

# Save processed validation data
processed_val_data_path = os.path.join(dataset_folder, "processed_val_data.csv")
processed_val_df.to_csv(processed_val_data_path, index=False)

print(f"\nProcessed Validation Data Saved at: {processed_val_data_path}")

# Display first 5 rows of processed validation data
print("\nProcessed Validation Data Preview:")
print(processed_val_df.head())
