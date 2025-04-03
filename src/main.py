#pip install pandas openpyxl scikit-learn matplotlib seaborn

import pandas as pd

# Define file paths (adjust if necessary)
train_path = "D:\\Users\\amisa\\Downloads\\Social Media Usage - Train.xlsm"
test_path = "D:\\Users\\amisa\\Downloads\\Social Media Usage - Test.xlsm"
val_path = "D:\\Users\\amisa\\Downloads\\Social Media Usage - Val.xlsm"
sleep_path = "D:\\Users\\amisa\\Downloads\\Sleep Dataset.xlsm"

# Load datasets
train_df = pd.read_excel(train_path)
test_df = pd.read_excel(test_path)
val_df = pd.read_excel(val_path)
sleep_df = pd.read_excel(sleep_path, sheet_name="Sleep Dataset")

# Print basic info
print("Train Dataset:", train_df.shape)
print("Test Dataset:", test_df.shape)
print("Validation Dataset:", val_df.shape)
print("Sleep Dataset:", sleep_df.shape)

print(val_df.dtypes)


# Identify numerical and categorical columns correctly
num_cols = train_df.select_dtypes(include=["number"]).columns.tolist()
cat_cols = train_df.select_dtypes(include=["object"]).columns.tolist()

print("Numerical Columns:", num_cols)
print("Categorical Columns:", cat_cols)

# Fill missing numerical values with median
train_df[num_cols] = train_df[num_cols].fillna(train_df[num_cols].median())
test_df[num_cols] = test_df[num_cols].fillna(test_df[num_cols].median())
val_df[num_cols] = val_df[num_cols].fillna(val_df[num_cols].median())

# Fill missing categorical values with mode
for col in cat_cols:
    train_df[col] = train_df[col].fillna(train_df[col].mode()[0])
    test_df[col] = test_df[col].fillna(test_df[col].mode()[0])
    val_df[col] = val_df[col].fillna(val_df[col].mode()[0])

print("Missing Values After Cleaning:")
print(train_df.isnull().sum())

