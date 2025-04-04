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
    usage_df = usage_df.dropna().drop(columns=["User_ID"], errors="ignore")
    usage_df = usage_df[usage_df["Gender"].isin(["Male", "Female"])]
    usage_df["Age"] = pd.to_numeric(usage_df["Age"])

    merged_data = []

    for gender in ["Male", "Female"]:
        usage_subset = usage_df[usage_df["Gender"] == gender]
        sleep_subset = sleep_df_augmented[sleep_df_augmented["Gender"] == gender]

        age_tree = KDTree(sleep_subset[["Age"]])
        nearest_indices = age_tree.query(usage_subset[["Age"]], k=3)[1]

        matched_sleep_data = []
        for indices in nearest_indices:
            idx = random.choice(indices)
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
    final_merged_df = final_merged_df.loc[:, ~final_merged_df.columns.duplicated()]

    # --- Feature Engineering ---
    required_columns = final_merged_df.columns

    # Ensure numeric conversions safely
    if "Daily_Usage_Time (minutes)" in required_columns:
        final_merged_df["Daily_Usage_Time (minutes)"] = pd.to_numeric(final_merged_df["Daily_Usage_Time (minutes)"],
                                                                      errors="coerce")

    if "Sleep Duration" in required_columns:
        final_merged_df["Sleep Duration"] = pd.to_numeric(final_merged_df["Sleep Duration"], errors="coerce")

    if "Stress Level" in required_columns:
        final_merged_df["Stress Level"] = pd.to_numeric(final_merged_df["Stress Level"], errors="coerce")

    if "Heart Rate" in required_columns:
        final_merged_df["Heart Rate"] = pd.to_numeric(final_merged_df["Heart Rate"], errors="coerce")

    if "Blood Pressure" in required_columns:
        final_merged_df["Blood Pressure"] = pd.to_numeric(final_merged_df["Blood Pressure"], errors="coerce")

    # Engagement Score
    if all(col in required_columns for col in
           ["Posts_Per_Day", "Likes_Received_Per_Day", "Comments_Received_Per_Day", "Messages_Sent_Per_Day"]):
        final_merged_df["Engagement_Score"] = (
                final_merged_df["Posts_Per_Day"] +
                final_merged_df["Likes_Received_Per_Day"] +
                final_merged_df["Comments_Received_Per_Day"] +
                final_merged_df["Messages_Sent_Per_Day"]
        )

    # Addiction Score
    if "Engagement_Score" in final_merged_df and "Daily_Usage_Time (minutes)" in final_merged_df:
        final_merged_df["Addiction_Score"] = (
                final_merged_df["Engagement_Score"] * final_merged_df["Daily_Usage_Time (minutes)"] / 100
        )

    # Sleep Impact Score
    if "Daily_Usage_Time (minutes)" in final_merged_df and "Sleep Duration" in final_merged_df:
        final_merged_df["Sleep_Impact_Score"] = (
                final_merged_df["Daily_Usage_Time (minutes)"] / final_merged_df["Sleep Duration"]
        )

    # Stress Usage Index
    if "Stress Level" in final_merged_df and "Daily_Usage_Time (minutes)" in final_merged_df:
        final_merged_df["Stress_Usage_Index"] = (
                final_merged_df["Stress Level"] * final_merged_df["Daily_Usage_Time (minutes)"]
        )

    # BMI Mapping and Health Burden Score
    if "BMI Category" in final_merged_df:
        bmi_mapping = {"Underweight": 0, "Normal": 1, "Overweight": 2, "Obese": 3}
        final_merged_df["BMI_Code"] = final_merged_df["BMI Category"].map(bmi_mapping)

    if all(col in final_merged_df for col in ["Addiction_Score", "Heart Rate", "Blood Pressure", "BMI_Code"]):
        final_merged_df["Health_Burden_Score"] = (
                final_merged_df["Addiction_Score"] * (
                final_merged_df["Heart Rate"] +
                final_merged_df["Blood Pressure"] +
                final_merged_df["BMI_Code"]
        )
        )


# --- Merge and Save All ---
merge_and_save(train_df, sleep_df_augmented, train_output)
merge_and_save(test_df, sleep_df_augmented, test_output)
merge_and_save(val_df, sleep_df_augmented, val_output)

# --- Verification ---
train_df = pd.read_csv(train_output)
print("===== 📊 Merged Train Dataset =====")
print(train_df.head())

print("\n--- Summary Statistics ---")
print(train_df.describe(include='all'))

print("\n--- Missing Values ---")
print(train_df.isnull().sum())
