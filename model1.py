import pandas as pd
from scipy.spatial import KDTree
import random

# Load datasets
sleep_file = "D:\\Sanika\\Datathon\\byte-belles\\Data\\Sleep Dataset.xlsm"
usage_file = "D:\\Sanika\\Datathon\\byte-belles\\Data\\Social Media Usage - Train.xlsx"

sleep_df = pd.read_excel(sleep_file, sheet_name="Sleep Dataset")
usage_df = pd.read_excel(usage_file)

# Step 1: Clean Social Media Usage Dataset
usage_df_cleaned = usage_df.dropna().drop(columns=["User_ID"])  # Remove null rows and drop User_ID
usage_df_cleaned = usage_df_cleaned[usage_df_cleaned["Gender"].isin(["Male", "Female"])]  # Remove non-binary gender

# Step 2: Clean and Augment Sleep Dataset
sleep_df_cleaned = sleep_df.drop(columns=["Person ID"])  # Remove Person ID
sleep_df_cleaned = sleep_df_cleaned[sleep_df_cleaned["Gender"].isin(["Male", "Female"])]  # Remove non-binary gender

# Data augmentation: Duplicate sleep dataset to double its size
sleep_df_augmented = pd.concat([sleep_df_cleaned, sleep_df_cleaned.copy()], ignore_index=True)

# Convert age columns to numeric
usage_df_cleaned["Age"] = pd.to_numeric(usage_df_cleaned["Age"])
sleep_df_augmented["Age"] = pd.to_numeric(sleep_df_augmented["Age"])

# Step 3: Nearest Age Matching with Upsampling (Random Selection)
merged_data = []

for gender in ["Male", "Female"]:
    # Filter datasets by gender
    usage_subset = usage_df_cleaned[usage_df_cleaned["Gender"] == gender]
    sleep_subset = sleep_df_augmented[sleep_df_augmented["Gender"] == gender]

    # Build a KDTree for nearest neighbor search based on age
    age_tree = KDTree(sleep_subset[["Age"]])

    # Find nearest ages for each user in the usage dataset
    nearest_indices = age_tree.query(usage_subset[["Age"]])[1]

    # Instead of directly assigning the nearest match, randomly select from close matches
    matched_sleep_data = []
    for idx in nearest_indices:
        # Get all sleep records with the closest age
        closest_age = sleep_subset.iloc[idx]["Age"]
        close_matches = sleep_subset[sleep_subset["Age"] == closest_age]

        # Randomly select one row from available close matches
        matched_sleep_data.append(close_matches.sample(n=1, random_state=random.randint(1, 1000)).iloc[0])

    # Convert list of matched sleep data to DataFrame
    matched_sleep_df = pd.DataFrame(matched_sleep_data).reset_index(drop=True)

    # Combine the matched data with the original social media usage data
    merged_subset = pd.concat([usage_subset.reset_index(drop=True), matched_sleep_df], axis=1)
    merged_data.append(merged_subset)

# Combine male and female merged data
final_merged_df = pd.concat(merged_data, ignore_index=True)

# Step 4: Save the new merged dataset
output_file = "Merged_Dataset_Upsampled.csv"
final_merged_df.to_csv(output_file, index=False)

print(f"Merged dataset saved as {output_file}")
