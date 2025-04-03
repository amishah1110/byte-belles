import pandas as pd

# Load datasets
train_df = pd.read_excel("C:\\Users\\ishit\\PycharmProjects\\datathon 2025\\data\\Social Media Usage - Train.xlsm", engine="openpyxl")
test_df = pd.read_excel("C:\\Users\\ishit\\PycharmProjects\\datathon 2025\\data\\Social Media Usage - Test.xlsm", engine="openpyxl")
val_df = pd.read_excel("C:\\Users\\ishit\\PycharmProjects\\datathon 2025\\data\\Social Media Usage - Val.xlsm", engine="openpyxl")
sleep_df = pd.read_excel("C:\\Users\\ishit\\PycharmProjects\\datathon 2025\\data\\Sleep Dataset.xlsm", engine="openpyxl")

# ✅ Convert 'Age' to numeric (fix the error)
def safe_convert_age(age):
    try:
        return int(age)
    except:
        return None  # If conversion fails, return None

for df in [train_df, test_df, val_df, sleep_df]:
    df['Age'] = df['Age'].apply(safe_convert_age)  # Convert Age to int

# ✅ Convert 'Age' to 'Age_Group' in ALL datasets
def categorize_age(age):
    if pd.isna(age):
        return 'Unknown'
    elif age < 18:
        return 'Under 18'
    elif age <= 25:
        return '18-25'
    elif age <= 35:
        return '26-35'
    elif age <= 50:
        return '36-50'
    else:
        return '50+'

for df in [train_df, test_df, val_df, sleep_df]:
    df['Age_Group'] = df['Age'].apply(categorize_age)  # Create Age_Group
    df.drop(columns=['Age'], inplace=True)  # Drop 'Age' after converting

# ✅ Merge Train, Test, and Validation datasets with Sleep Data using 'Age_Group' & 'Gender'
train_merged = pd.merge(train_df, sleep_df, on=['Age_Group', 'Gender'], how='left')
test_merged = pd.merge(test_df, sleep_df, on=['Age_Group', 'Gender'], how='left')
val_merged = pd.merge(val_df, sleep_df, on=['Age_Group', 'Gender'], how='left')

# Drop unnecessary columns (if needed)
columns_to_drop = ['User_ID', 'Person ID']
for df in [train_merged, test_merged, val_merged]:
    df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True, errors='ignore')

# 🚨 Remove all rows with any missing values
train_merged.dropna(inplace=True)
test_merged.dropna(inplace=True)
val_merged.dropna(inplace=True)


# Save the cleaned datasets
train_merged.to_csv("C:\\Users\\ishit\\PycharmProjects\\datathon 2025\\data\\cleaned_train.csv", index=False)
test_merged.to_csv("C:\\Users\\ishit\\PycharmProjects\\datathon 2025\\data\\cleaned_test.csv", index=False)
val_merged.to_csv("C:\\Users\\ishit\\PycharmProjects\\datathon 2025\\data\\cleaned_validation.csv", index=False)

print("✅ Data Merging Complete! No errors, go sleep peacefully. 😴💙")

# Verification: Show top 5 rows and dataset summary
for name, df in zip(["Train", "Test", "Validation"], [train_merged, test_merged, val_merged]):
    print(f"\n===== {name} Dataset =====")
    print(df.head())  # Show first 5 rows
    print("\n--- Summary Statistics ---")
    print(df.describe(include='all'))  # Show statistics
    print("\n--- Missing Values ---")
    print(df.isnull().sum())  # Show missing values count
