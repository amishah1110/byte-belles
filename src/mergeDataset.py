import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import random

# --- File Paths ---
data_dir = "C:\\Users\\ishit\\PycharmProjects\\datathon 2025\\data"
output_dir = "C:\\Users\\ishit\\PycharmProjects\\datathon 2025\\data\\MergedData"

sleep_file = f"{data_dir}\\Sleep Dataset.xlsm"
train_file = f"{data_dir}\\Social Media Usage - Train.xlsm"
test_file = f"{data_dir}\\Social Media Usage - Test.xlsm"
val_file = f"{data_dir}\\Social Media Usage - Val.xlsm"

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

    # Convert numeric columns safely
    final_merged_df["Daily_Usage_Time (minutes)"] = pd.to_numeric(final_merged_df["Daily_Usage_Time (minutes)"], errors="coerce")
    final_merged_df["Sleep Duration"] = pd.to_numeric(final_merged_df["Sleep Duration"], errors="coerce")
    final_merged_df["Stress Level"] = pd.to_numeric(final_merged_df["Stress Level"], errors="coerce")

    # --- Feature Engineering ---
    # 1. Engagement Score
    final_merged_df["Engagement_Score"] = (
            final_merged_df["Posts_Per_Day"] +
            final_merged_df["Likes_Received_Per_Day"] +
            final_merged_df["Comments_Received_Per_Day"] +
            final_merged_df["Messages_Sent_Per_Day"]
    )

    # 2. Addiction Score
    final_merged_df["Addiction_Score"] = (
            final_merged_df["Engagement_Score"] * final_merged_df["Daily_Usage_Time (minutes)"] / 100
    )

    # 3. Sleep Impact Score
    final_merged_df["Sleep_Impact_Score"] = (
            final_merged_df["Daily_Usage_Time (minutes)"] / final_merged_df["Sleep Duration"]
    )

    # 4. Stress Usage Index
    final_merged_df["Stress_Usage_Index"] = (
            final_merged_df["Stress Level"] * final_merged_df["Daily_Usage_Time (minutes)"]
    )

    # 5. Health Burden Score
    bmi_mapping = {"Underweight": 0, "Normal": 1, "Overweight": 2, "Obese": 3}
    final_merged_df["BMI_Code"] = final_merged_df["BMI Category"].map(bmi_mapping)

    # Convert required health columns to numeric safely
    final_merged_df["Heart Rate"] = pd.to_numeric(final_merged_df["Heart Rate"], errors="coerce")
    final_merged_df["Blood Pressure"] = pd.to_numeric(final_merged_df["Blood Pressure"], errors="coerce")
    final_merged_df["BMI_Code"] = pd.to_numeric(final_merged_df["BMI_Code"], errors="coerce")

    final_merged_df["Health_Burden_Score"] = (
            final_merged_df["Addiction_Score"] * (
            final_merged_df["Heart Rate"] +
            final_merged_df["Blood Pressure"] +
            final_merged_df["BMI_Code"]
    )
    )

    # Save to CSV
    final_merged_df.to_csv(output_path, index=False)
    print(f"✅ Merged file saved at: {output_path}")

    # --- Data Visualization ---
    sns.scatterplot(data=final_merged_df, x="Addiction_Score", y="Sleep Duration", hue="Gender")
    plt.title("Addiction Score vs Sleep Duration")
    plt.xlabel("Addiction Score")
    plt.ylabel("Sleep Duration (hours)")
    plt.show()

    sns.boxplot(data=final_merged_df, x="Stress Level", y="Engagement_Score", palette="Reds")
    plt.title("Engagement Score across Stress Levels")
    plt.xlabel("Stress Level (1 = Low, 5 = High)")
    plt.ylabel("Engagement Score")
    plt.show()

    # sns.violinplot(data=final_merged_df, x="Platform", y="Health_Burden_Score", palette="muted")
    # plt.title("Health Burden Score across Platforms")
    # plt.xticks(rotation=45)
    # plt.show()

    sns.countplot(data=final_merged_df, x="Dominant_Emotion", hue="Sleep Disorder", palette="Set2")
    plt.title("Sleep Disorder Occurrence by Dominant Emotion")
    plt.xticks(rotation=45)
    plt.show()

    sns.boxplot(data=final_merged_df, x="Occupation", y="Stress Level", palette="pastel")
    plt.title("Stress Level Distribution by Occupation")
    plt.xticks(rotation=45)
    plt.show()

    # sns.scatterplot(data=final_merged_df, x="Daily Steps", y="Health_Burden_Score", hue="Gender")
    # plt.title("Health Burden Score vs Daily Steps")
    # plt.xlabel("Daily Steps")
    # plt.ylabel("Health Burden Score")
    # plt.show()

    return final_merged_df


# --- Merge and Save All ---
merge_and_save(train_df, sleep_df_augmented, train_output)
# merge_and_save(test_df, sleep_df_augmented, test_output)
# Verification: Show top 5 rows and dataset summary
train_df = pd.read_csv(train_output)
print("===== 📊 Merged Train Dataset =====")
print(train_df.head())  # Top 5 rows

print("\n--- Summary Statistics ---")
print(train_df.describe(include='all'))  # All-column summary

print("\n--- Missing Values ---")
print(train_df.isnull().sum())  # Count of nulls per column
# merge_and_save(val_df, sleep_df_augmented, val_output)
