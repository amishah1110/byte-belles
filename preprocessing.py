import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
file_path = "D:\\Datathon\\byte-belles\\Data\\Sleep Dataset.xlsm.xlsx"  # Update with your actual file path
df = pd.read_excel(file_path)

# Split 'Blood Pressure' into 'Systolic BP' and 'Diastolic BP'
df[['Systolic BP', 'Diastolic BP']] = df['Blood Pressure'].str.split('/', expand=True).astype(float)

# Drop the original 'Blood Pressure' column
df.drop(columns=['Blood Pressure'], inplace=True)

# Encode categorical variables
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})  # Binary encoding
df['BMI Category'] = df['BMI Category'].astype('category').cat.codes  # Convert categorical to numerical
df['Sleep Disorder'] = df['Sleep Disorder'].astype('category').cat.codes  # Convert categorical to numerical
df['Occupation'] = df['Occupation'].astype('category').cat.codes  # Convert categorical to numerical

# Normalize numerical features using Min-Max Scaling
scaler = MinMaxScaler()
num_cols = ['Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 'Stress Level',
            'Heart Rate', 'Daily Steps', 'Systolic BP', 'Diastolic BP']
df[num_cols] = scaler.fit_transform(df[num_cols])
#df.to_csv("preprocessed_sleep_data.csv", index=False)
print(df)  # Prints the full DataFrame

# Exploratory Data Analysis (EDA)
# Correlation heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Features")
plt.show()

# Boxplot for sleep duration across different age groups
plt.figure(figsize=(10, 5))
sns.boxplot(x=df["Age"], y=df["Sleep Duration"], palette="coolwarm")
plt.title("Sleep Duration Across Different Age Groups")
plt.xlabel("Age")
plt.ylabel("Normalized Sleep Duration")
plt.xticks(rotation=45)
plt.show()

# Scatter plot for Stress Level vs. Physical Activity Level
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df["Physical Activity Level"], y=df["Stress Level"], alpha=0.6, color="red")
plt.title("Stress Level vs. Physical Activity Level")
plt.xlabel("Normalized Physical Activity Level")
plt.ylabel("Normalized Stress Level")
plt.show()
