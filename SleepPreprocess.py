import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
file_path = "D:\\Sanika\\Datathon\\byte-belles\\Data\\Sleep Dataset.xlsm"  # Update with your actual file path
df = pd.read_excel(file_path)

# Split 'Blood Pressure' into 'Systolic BP' and 'Diastolic BP'
df[['Systolic BP', 'Diastolic BP']] = df['Blood Pressure'].str.split('/', expand=True).astype(float)

# Drop the original 'Blood Pressure' column
df.drop(columns=['Blood Pressure'], inplace=True)

# Drop 'Person ID' and move it to the last column
if 'Person ID' in df.columns:
    person_id = df.pop('Person ID')  # Remove 'Person ID' from the dataset

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

# Feature Selection - Assuming 'Sleep Disorder' is the target variable
X = df.drop(columns=['Sleep Disorder', 'Systolic BP', 'Diastolic BP'])  # Excluding BP
y = df['Sleep Disorder']

# Train a simple RandomForest model for feature importance
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Extract feature importances
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, hue='Feature', palette='magma', legend=False)
plt.title('Feature Importance Ranking')
plt.show()
# Print top features
print(feature_importance_df.head(10))