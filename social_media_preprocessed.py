import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

# Define dataset folder
dataset_folder = "D:\\Datathon\\byte-belles\\Data"

# Define file paths dynamically
file_paths = {
    "train": os.path.join(dataset_folder, "Social Media Usage - Train.xlsm.xlsx"),
    "test": os.path.join(dataset_folder, "Social Media Usage - Test.xlsm.xlsx"),
    "val": os.path.join(dataset_folder, "Social Media Usage - Val.xlsm.xlsx"),
    "sleep": os.path.join(dataset_folder, "Sleep Dataset.xlsm.xlsx")
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
    (processed_df["Daily_Usage_Time (minutes)"] * 0.3) +
    (processed_df["Posts_Per_Day"] * 0.05) +
    (processed_df["Likes_Received_Per_Day"] * 0.4) +
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


# Encode categorical features using OrdinalEncoder
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# Dictionary to store categorical mappings
encoding_maps = {}

# Ensure categorical columns are strings before encoding
for col in ["Gender", "Platform", "Dominant_Emotion", "Age_Group"]:
    if col in processed_df.columns:
        processed_df[col] = processed_df[col].astype(str).fillna("Unknown")

        # Fit encoder and store mappings
        encoder.fit(processed_df[[col]])
        encoding_maps[col] = {category: idx for idx, category in enumerate(encoder.categories_[0])}

        # Transform data
        processed_df[col] = encoder.transform(processed_df[[col]])

print("\nCategory Encodings:")
for col, mapping in encoding_maps.items():
    print(f"{col}: {mapping}")

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
sns.set(style="whitegrid")

### Reverse Mapping for Encoded Features
def decode_column(df, column, mapping_dict):
    return df[column].map({v: k for k, v in mapping_dict.items()})

### Apply Reverse Mapping to Processed Data
processed_df['Gender'] = decode_column(processed_df, 'Gender', encoding_maps['Gender'])
processed_df['Platform'] = decode_column(processed_df, 'Platform', encoding_maps['Platform'])
processed_df['Dominant_Emotion'] = decode_column(processed_df, 'Dominant_Emotion', encoding_maps['Dominant_Emotion'])
processed_df['Age_Group'] = decode_column(processed_df, 'Age_Group', encoding_maps['Age_Group'])

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Age Group Distribution
processed_df['Age_Group'].value_counts().plot.pie(
    autopct='%1.1f%%', startangle=90, cmap='coolwarm', ax=axes[0], fontsize=12
)
axes[0].set_title("Age Group Distribution", fontsize=14)
axes[0].set_ylabel('')

# Platform Usage
processed_df['Platform'].value_counts().plot.pie(
    autopct='%1.1f%%', startangle=90, cmap='coolwarm', ax=axes[1], fontsize=12
)
axes[1].set_title("Platform Usage", fontsize=14)
axes[1].set_ylabel('')

# Adjust layout
plt.tight_layout()
plt.show()
#Platform Preference by Age Group (Processed Data)
plt.figure(figsize=(10, 5))
sns.countplot(data=processed_df, x='Platform', hue='Age_Group', order=processed_df['Platform'].value_counts().index)
plt.title("Platform Preference by Age Group (Processed Data)")
plt.xticks(rotation=45)
plt.show()

#Engagement Score vs Age Group
plt.figure(figsize=(10, 5))
sns.boxplot(data=processed_df, x='Age_Group', y='Engagement_Score')
plt.title("Engagement Score by Age Group")
plt.xticks(rotation=45)
plt.show()

#Engagement Score vs Emotion
plt.figure(figsize=(10, 5))
sns.boxplot(data=processed_df, x='Dominant_Emotion', y='Engagement_Score')
plt.title("Engagement Score by Emotion")
plt.xticks(rotation=45)
plt.show()

#Platform vs Emotion Correlation
plt.figure(figsize=(10, 6))
platform_emotion_correlation = pd.crosstab(train_df['Platform'], train_df['Dominant_Emotion'])
sns.heatmap(platform_emotion_correlation, cmap='Blues', annot=True, fmt='d')
plt.title("Platform vs Emotion Correlation")
plt.show()

#Boxplot - Daily Usage Time by Platform
plt.figure(figsize=(12, 5))
sns.boxplot(data=processed_df, x='Platform', y='Daily_Usage_Time (minutes)')
plt.title("Daily Usage Time by Platform")
plt.xticks(rotation=45)
plt.show()



print("\nData Visualization Completed.")

# Selecting the most significant feature: Engagement_Score
feature = "Daily_Usage_Time (minutes)"

# Calculate statistics
std_dev = train_df[feature].std()
mean_value = train_df[feature].mean()
variance = train_df[feature].var()

# Prepare data for plotting
plot_data = train_df[feature].sort_values().reset_index(drop=True)

plt.figure(figsize=(10, 5))

# Line plot for Engagement_Score
plt.plot(plot_data, label=f"{feature} (Sorted)", color='royalblue', linewidth=2)

# Horizontal lines for Mean & Standard Deviation
plt.axhline(y=mean_value, color='green', linestyle='--', linewidth=2, label=f"Mean: {mean_value:.2f}")
plt.axhline(y=mean_value + std_dev, color='orange', linestyle='--', linewidth=2, label=f"Mean + 1 SD: {mean_value + std_dev:.2f}")
plt.axhline(y=mean_value - std_dev, color='orange', linestyle='--', linewidth=2, label=f"Mean - 1 SD: {mean_value - std_dev:.2f}")

# Title and Labels
plt.title(f"Analysis of {feature}: Mean, Standard Deviation, and Variance", fontsize=14)
plt.xlabel("Sorted Data Index", fontsize=12)
plt.ylabel(feature, fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.show()

print(f"\n📊 {feature} Analysis:")
print(f"Mean: {mean_value:.2f}")
print(f"Standard Deviation: {std_dev:.2f}")
print(f"Variance: {variance:.2f}")