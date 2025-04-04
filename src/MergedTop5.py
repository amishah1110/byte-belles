#MergedTOP5

import pandas as pd
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
train_output = f"{output_dir}\\Merged_Train_Top5.csv"
test_output = f"{output_dir}\\Merged_Test_Top5.csv"
val_output = f"{output_dir}\\Merged_Val_Top5.csv"

# --- Load datasets ---
sleep_df = pd.read_excel(sleep_file, sheet_name="Sleep Dataset")
train_df = pd.read_excel(train_file)
test_df = pd.read_excel(test_file)
val_df = pd.read_excel(val_file)

# --- Clean and filter sleep data ---
sleep_df = sleep_df.drop(columns=["Person ID"], errors="ignore")
sleep_df = sleep_df[sleep_df["Gender"].isin(["Male", "Female"])]
sleep_df["Age"] = pd.to_numeric(sleep_df["Age"])

# Keep only top 5 features
top_features = ["BMI Category", "Occupation", "Age", "Sleep Duration", "Physical Activity Level", "Gender"]
sleep_df = sleep_df[top_features]

# Duplicate dataset for augmentation
sleep_df_augmented = pd.concat([sleep_df, sleep_df.copy()], ignore_index=True)


# --- Function to merge datasets based on Age & Gender ---
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
print("===== 📊 Merged Train Dataset (Top 5 Features) =====")
print(train_df.head())  # Top 5 rows

print("\n--- Summary Statistics ---")
print(train_df.describe(include='all'))  # All-column summary

print("\n--- Missing Values ---")
print(train_df.isnull().sum())  # Count of nulls per column

#Post merging
import pandas as pd

# === Load merged dataset ===
merged_path = "C:\\Users\\ishit\\PycharmProjects\\datathon 2025\\data\\MergedData\\Merged_Train_Top5.csv"  # Use other paths for test/val
df = pd.read_csv(merged_path)

# === Feature Engineering ===

# 1. Social Media Addiction Score (SMAS)
# Normalize relevant usage features and calculate average
usage_features = ['Daily_Usage_Time (minutes)', 'Posts_Per_Day', 'Likes_Received_Per_Day',
                  'Comments_Received_Per_Day', 'Messages_Sent_Per_Day']

df['Social Media Addiction Score'] = df[usage_features].mean(axis=1)

# 2. Stress Impact Score
# Assuming stress manifests via low physical activity + high social media usage
df['Stress Impact Score'] = (df['Social Media Addiction Score']) / (df['Physical Activity Level'] + 1)

# 3. Stress Usage Index (SUI)
# Combine high daily usage and low sleep
df['Stress Usage Index'] = (df['Daily_Usage_Time (minutes)']) / (df['Sleep Duration'] + 1)

# Save engineered dataset
output_path = merged_path.replace(".csv", "_engineered.csv")
df.to_csv(output_path, index=False)
print(f"✅ Feature-engineered dataset saved to: {output_path}")

# Preview
print(df.head())

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load engineered dataset
file_path = "C:\\Users\\ishit\\PycharmProjects\\datathon 2025\\data\\MergedData\\Merged_Train_Top5_engineered.csv"
df = pd.read_csv(file_path)

sns.set(style="whitegrid", palette="muted")

# 1. Distribution of Social Media Addiction Score
plt.figure(figsize=(8, 5))
sns.histplot(df['Social Media Addiction Score'], bins=30, kde=True, color='skyblue')
plt.title("Distribution of Social Media Addiction Score")
plt.xlabel("Social Media Addiction Score")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 2. Stress Impact Score vs Physical Activity Level
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Physical Activity Level', y='Stress Impact Score', hue='Gender', data=df)
plt.title("Stress Impact Score vs Physical Activity Level")
plt.xlabel("Physical Activity Level")
plt.ylabel("Stress Impact Score")
plt.tight_layout()
plt.show()

# 3. Stress Usage Index vs Sleep Duration
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Sleep Duration', y='Stress Usage Index', hue='Gender', data=df)
plt.title("Stress Usage Index vs Sleep Duration")
plt.xlabel("Sleep Duration (hours)")
plt.ylabel("Stress Usage Index")
plt.tight_layout()
plt.show()

# 4. Boxplot of Social Media Addiction Score by Occupation
plt.figure(figsize=(12, 5))
sns.boxplot(x='Occupation', y='Social Media Addiction Score', data=df)
plt.title("Addiction Score by Occupation")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 5. Correlation Heatmap of new features with health-related ones
plt.figure(figsize=(10, 6))
corr_cols = ['Social Media Addiction Score', 'Stress Impact Score', 'Stress Usage Index',
             'Sleep Duration', 'Physical Activity Level']
sns.heatmap(df[corr_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap: Social Media Usage vs Health Metrics")
plt.tight_layout()
plt.show()

# === Feature Engineering for Test Set ===
test_path = "C:\\Users\\ishit\\PycharmProjects\\datathon 2025\\data\\MergedData\\Merged_Test_Top5.csv"
df_test = pd.read_csv(test_path)

# 1. Social Media Addiction Score (SMAS)
usage_features = ['Daily_Usage_Time (minutes)', 'Posts_Per_Day', 'Likes_Received_Per_Day',
                  'Comments_Received_Per_Day', 'Messages_Sent_Per_Day']
df_test['Social Media Addiction Score'] = df_test[usage_features].mean(axis=1)

# 2. Stress Impact Score
df_test['Stress Impact Score'] = (df_test['Social Media Addiction Score']) / (df_test['Physical Activity Level'] + 1)

# 3. Stress Usage Index
df_test['Stress Usage Index'] = (df_test['Daily_Usage_Time (minutes)']) / (df_test['Sleep Duration'] + 1)

# Save engineered test dataset
test_output_path = test_path.replace(".csv", "_engineered.csv")
df_test.to_csv(test_output_path, index=False)
print(f"✅ Feature-engineered test dataset saved to: {test_output_path}")