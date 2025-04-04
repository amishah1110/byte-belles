import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

# Define dataset folder
dataset_folder = "D:\\Datathon\\byte-belles\\Data"

# Load the Social Media Usage training dataset
file_path = os.path.join(dataset_folder, "Social Media Usage - Train.xlsm.xlsx")
social_df = pd.read_excel(file_path)

# Remove blank rows
social_df.dropna(how='all', inplace=True)
social_df.reset_index(drop=True, inplace=True)

# Relevant numerical and categorical columns
num_cols = ["Daily_Usage_Time (minutes)", "Posts_Per_Day", "Likes_Received_Per_Day",
            "Comments_Received_Per_Day", "Messages_Sent_Per_Day"]

cat_cols = ["User_ID", "Age", "Gender", "Platform", "Dominant_Emotion"]

# Convert numerical columns to float
for col in num_cols:
    social_df[col] = pd.to_numeric(social_df[col], errors='coerce')

# Fill missing numerical values with median
social_df[num_cols] = social_df[num_cols].apply(lambda x: x.fillna(x.median()), axis=0)

# Convert categorical columns to string
for col in cat_cols:
    social_df[col] = social_df[col].astype(str).fillna("Unknown")

# **🔹 Standardizing Gender to Match Sleep Dataset (0: Male, 1: Female)**
gender_mapping = {'Male': 0, 'Female': 1, 'Non-binary': 2}  # Keep 'Non-binary' as separate category
social_df["Gender"] = social_df["Gender"].map(gender_mapping).fillna(2)  # Default to 'Non-binary'

# Convert 'Age' to numeric, forcing errors to NaN (to handle incorrect entries like 'Male')
social_df["Age"] = pd.to_numeric(social_df["Age"], errors='coerce')

# Function to categorize age groups
def age_grouping(age):
    if pd.isna(age): return "Unknown"  # Handle NaN values safely
    age = int(age)
    return "Under 18" if age < 18 else "18-24" if age < 25 else "25-34" if age < 35 else "35-44" if age < 45 else "45-59" if age < 60 else "60+"

# Apply age grouping function
social_df["Age Group"] = social_df["Age"].apply(age_grouping)

print(social_df[["Age", "Age Group"]].head())  # Debugging step to verify

# **🔹 Convert Age into Age Groups**
def age_grouping(age):
    if pd.isna(age): return "Unknown"
    age = int(age)
    return "Under 18" if age < 18 else "18-24" if age < 25 else "25-34" if age < 35 else "35-44" if age < 45 else "45-59" if age < 60 else "60+"
social_df["Age Group"] = social_df["Age"].apply(age_grouping)

# **🔹 Ensure 'Age Group' is a string (matches Sleep Dataset)**
social_df["Age Group"] = social_df["Age Group"].astype(str)

# **🔹 Encoding 'Age Group' with OrdinalEncoder**
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
social_df[["Age Group"]] = encoder.fit_transform(social_df[["Age Group"]])

# **🔹 Standardizing & Scaling Numerical Features**
scaler = StandardScaler()
social_df[num_cols] = scaler.fit_transform(social_df[num_cols])

# Save the processed data
processed_data_path = os.path.join(dataset_folder, "processed_social_data.csv")
social_df.to_csv(processed_data_path, index=False)
print(f"\n✅ Processed Social Media Data Saved at: {processed_data_path}")

# Print the first few rows
print("\nProcessed Social Media Data Sample:")
print(social_df.head())
