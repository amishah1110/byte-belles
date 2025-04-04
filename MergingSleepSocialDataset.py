import pandas as pd
from scipy.spatial import KDTree
import random

# --- File Paths ---
data_dir = "D:\\Sanika\\Datathon\\byte-belles\\Data"
output_dir = "D:\\Sanika\\Datathon\\byte-belles\\Data\\MergedData"

sleep_file = f"{data_dir}\\Sleep Dataset.xlsm"
train_file = f"{data_dir}\\Social Media Usage - Train.xlsx"
test_file = f"{data_dir}\\Social Media Usage - Test.xlsx"
val_file = f"{data_dir}\\Social Media Usage - Val.xlsx"

# Output files
train_output = f"{output_dir}\\Merged_Train.csv"
test_output = f"{output_dir}\\Merged_Test.csv"
val_output = f"{output_dir}\\Merged_Val.csv"

# --- Load datasets ---
sleep_df = pd.read_excel(sleep_file, sheet_name="Sleep Dataset")
train_df = pd.read_excel(train_file)
test_df = pd.read_excel(test_file)
val_df = pd.read_excel(val_file)

# --- Clean and augment sleep data ---
sleep_df_cleaned = sleep_df.drop(columns=["Person ID"], errors="ignore")
sleep_df_cleaned = sleep_df_cleaned[sleep_df_cleaned["Gender"].isin(["Male", "Female"])]
sleep_df_augmented = pd.concat([sleep_df_cleaned, sleep_df_cleaned.copy()], ignore_index=True)
sleep_df_augmented["Age"] = pd.to_numeric(sleep_df_augmented["Age"])


# --- Function to merge and save ---
def merge_and_save(usage_df, sleep_df_augmented, output_path):
    # Clean usage data
    usage_df = usage_df.dropna().drop(columns=["User_ID"], errors="ignore")
    usage_df = usage_df[usage_df["Gender"].isin(["Male", "Female"])]
    usage_df["Age"] = pd.to_numeric(usage_df["Age"])

    merged_data = []

    for gender in ["Male", "Female"]:
        usage_subset = usage_df[usage_df["Gender"] == gender]
        sleep_subset = sleep_df_augmented[sleep_df_augmented["Gender"] == gender]

        # Build KDTree for nearest neighbor search
        age_tree = KDTree(sleep_subset[["Age"]])
        nearest_indices = age_tree.query(usage_subset[["Age"]], k=3)[1]  # Get top 3 nearest ages

        matched_sleep_data = []
        for indices in nearest_indices:
            # Randomly choose one of the k nearest neighbors
            idx = random.choice(indices)

            # Ensure Age is an int
            closest_age = int(sleep_subset.iloc[idx]["Age"])
            close_matches = sleep_subset[sleep_subset["Age"].astype(int) == closest_age]

            if not close_matches.empty:
                matched_sleep_data.append(
                    close_matches.sample(n=1, random_state=random.randint(1, 1000)).iloc[0]
                )

        matched_sleep_df = pd.DataFrame(matched_sleep_data).reset_index(drop=True)
        merged_subset = pd.concat([usage_subset.reset_index(drop=True), matched_sleep_df], axis=1)
        merged_data.append(merged_subset)

    final_merged_df = pd.concat(merged_data, ignore_index=True)

    # Remove duplicate columns
    final_merged_df = final_merged_df.loc[:, ~final_merged_df.columns.duplicated()]

    # Save to CSV
    final_merged_df.to_csv(output_path, index=False)

    print(f"✅ Merged file saved at: {output_path}")
    return final_merged_df


# --- Merge and Save All ---
merge_and_save(train_df, sleep_df_augmented, train_output)
merge_and_save(test_df, sleep_df_augmented, test_output)
merge_and_save(val_df, sleep_df_augmented, val_output)

# Verification: Show top 5 rows and dataset summary
train_df = pd.read_csv(train_output)
print("===== 📊 Merged Train Dataset =====")
print(train_df.head())  # Top 5 rows

print("\n--- Summary Statistics ---")
print(train_df.describe(include='all'))  # All-column summary

print("\n--- Missing Values ---")
print(train_df.isnull().sum())  # Count of nulls per column