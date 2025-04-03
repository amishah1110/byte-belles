import pandas as pd
import os

# Define dataset folder
dataset_folder = "D:\\Users\\amisa\\Downloads"

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

# Print updated column names to debug
print("\nUpdated Column Names in Train Dataset:", train_df.columns.tolist())

# Identify numerical and categorical columns
num_cols = ["Daily_Usage_Time (minutes)", "Posts_Per_Day", "Likes_Received_Per_Day",
            "Comments_Received_Per_Day", "Messages_Sent_Per_Day"]

cat_cols = ["User_ID", "Age", "Gender", "Platform", "Dominant_Emotion"]

print("\nNumerical Columns:", num_cols)
print("Categorical Columns:", cat_cols)

# Convert numerical columns to float (checking existence before transformation)
for df in [train_df, test_df, val_df, sleep_df]:
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

# Apply preprocessing ONLY to training dataset
if not train_df.empty:
    train_df[num_cols] = train_df[num_cols].apply(lambda x: x.fillna(x.median()), axis=0)
    for col in cat_cols:
        if col in train_df.columns:
            train_df[col] = train_df[col].astype(str).fillna(train_df[col].mode()[0])

# Print missing values after cleaning
print("\nMissing Values After Cleaning (Train Only):")
print(train_df.isnull().sum())

print("\n✅ Data Preprocessing Completed Successfully (Only for Training Data)")

from sklearn.preprocessing import OrdinalEncoder

# Encode categorical features using OrdinalEncoder
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# Ensure categorical columns are strings and fill NaNs
for col in ["Gender", "Platform", "Dominant_Emotion"]:
    if col in train_df.columns:
        train_df[col] = train_df[col].astype(str).fillna("Unknown")

# Fit encoder on training data and transform datasets
if not train_df.empty:
    train_df[["Gender", "Platform", "Dominant_Emotion"]] = encoder.fit_transform(
        train_df[["Gender", "Platform", "Dominant_Emotion"]]
    )

print(train_df.head())
