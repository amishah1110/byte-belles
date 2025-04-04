import pandas as pd
from sklearn.utils import resample

# Load the datasets
sleep_data_path = "D:\\Datathon\\byte-belles\\Data\\Sleep Dataset.xlsm.xlsx"
social_media_data_path = "D:\\Datathon\\byte-belles\\Data\\Social Media Usage - Train.xlsm.xlsx"

sleep_df = pd.read_excel(sleep_data_path)
social_media_df = pd.read_excel(social_media_data_path)

# Drop irrelevant and missing values
sleep_df.drop(columns=["Person ID"], inplace=True, errors='ignore')
social_media_df.drop(columns=["User_ID"], inplace=True, errors='ignore')
social_media_df.dropna(subset=["Age", "Gender"], inplace=True)

# Convert Age to numeric and categorize into bins
age_bins = [18, 25, 35, 45, 55, 65, 100]
age_labels = ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]
sleep_df["Age Group"] = pd.cut(pd.to_numeric(sleep_df["Age"], errors='coerce'), bins=age_bins, labels=age_labels, right=False)
social_media_df["Age Group"] = pd.cut(pd.to_numeric(social_media_df["Age"], errors='coerce'), bins=age_bins, labels=age_labels, right=False)

# Drop original Age column
sleep_df.drop(columns=["Age"], inplace=True)
social_media_df.drop(columns=["Age"], inplace=True)

# Ensure Gender is properly formatted before grouping
social_media_df = social_media_df.dropna(subset=["Gender"])  # Drop NaNs
social_media_df["Gender"] = social_media_df["Gender"].astype(str).str.strip()  # Remove spaces

# Gender mapping
gender_mapping = {"Male": 0, "Female": 1}
social_media_df["Gender"] = social_media_df["Gender"].map(gender_mapping)

# Debugging: Check unique gender values
print("Unique Gender Values:", social_media_df["Gender"].unique())
print("Gender Data Type:", social_media_df["Gender"].dtype)

# Ensure no NaNs after mapping
social_media_df = social_media_df.dropna(subset=["Gender"])
social_media_df["Gender"] = social_media_df["Gender"].astype(int)

# Keep only numeric columns from social media dataset
numeric_cols = social_media_df.select_dtypes(include=['number']).columns.tolist()
social_media_df = social_media_df[numeric_cols + ["Age Group", "Gender"]]

# Function to handle safe resampling
def safe_resample(group, min_size):
    return resample(group, replace=True, n_samples=min(len(group), min_size), random_state=42)

# Balance the datasets via stratified resampling
min_samples = min(len(sleep_df), len(social_media_df))
sleep_balanced = sleep_df.groupby(["Age Group", "Gender"], group_keys=False).apply(lambda x: safe_resample(x, min_samples // 10))
social_media_balanced = social_media_df.groupby(["Age Group", "Gender"], group_keys=False).apply(lambda x: safe_resample(x, min_samples // 10))

# Merge datasets on Age Group and Gender
merged_df = pd.merge(sleep_balanced, social_media_balanced, on=["Age Group", "Gender"], how="inner")

# Save the final dataset
merged_df.to_csv("Merged_Health_SocialMedia.csv", index=False)

print("Preprocessing and merging completed. Data saved as 'Merged_Health_SocialMedia.csv'")
