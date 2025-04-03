import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns


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

# Compute correlation matrix
correlation_matrix = train_df[num_cols].corr()

# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
#plt.show()

# Find most related features (Top 3)
corr_unstacked = correlation_matrix.abs().unstack().sort_values(ascending=False)
corr_unstacked = corr_unstacked[corr_unstacked < 1]  # Remove self-correlations
top_correlated_features = corr_unstacked[:6].index.tolist()  # Get top 3 feature pairs

print("Most Related Features => Top 3 pairs based on correlation:")
for pair in top_correlated_features:
    print(f"{pair[0]} ↔ {pair[1]}: {correlation_matrix.loc[pair[0], pair[1]]:.2f}")

#KDE plot- Distribution of Daily Usage Time
plt.figure(figsize=(8, 5))
sns.histplot(train_df["Daily_Usage_Time (minutes)"], bins=30, kde=True, color="blue")
plt.xlabel("Daily Usage Time (minutes)")
plt.ylabel("Frequency")
plt.title("Distribution of Daily Usage Time")
#plt.show()



#-------------test preprocessing-------------#
# Identify columns that should be numeric and categorical
numeric_cols = ["Age", "Daily_Usage_Time (minutes)", "Posts_Per_Day",
                "Likes_Received_Per_Day", "Comments_Received_Per_Day", "Messages_Sent_Per_Day"]
categorical_cols = ["User_ID", "Gender", "Platform", "Dominant_Emotion"]

# Detect incorrect values and fix them
for col in numeric_cols:
    if col in test_df.columns:
        invalid_rows = test_df[test_df[col].apply(lambda x: not str(x).replace('.', '', 1).isdigit())]
        print(f"\n🚨 Incorrect values in '{col}':")
        print(invalid_rows[[col]].head())

        # Convert column to numeric, replacing invalid values with NaN
        test_df[col] = pd.to_numeric(test_df[col], errors="coerce")

        # Replace NaN with column mean
        test_df[col] = test_df[col].fillna(test_df[col].mean())

for col in categorical_cols:
    if col in test_df.columns:
        invalid_rows = test_df[test_df[col].apply(lambda x: str(x).replace('.', '', 1).isdigit())]
        print(f"\n🚨 Incorrect values in '{col}':")
        print(invalid_rows[[col]].head())

        # Convert column to string, replacing invalid numeric values with NaN
        test_df[col] = test_df[col].astype(str)

        # Replace invalid values with mode
        test_df[col] = test_df[col].apply(lambda x: x if not x.replace('.', '', 1).isdigit() else None)
        test_df[col] = test_df[col].fillna(test_df[col].mode()[0])

print("\n✅ Test Data Preprocessing Completed Successfully!")
