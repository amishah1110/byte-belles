import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
file_path = "C:\\Users\\ishit\\PycharmProjects\\datathon 2025\\data\\Sleep Dataset.xlsm"  # Update with your actual file path
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
# Exploratory Data Analysis (EDA)
# Correlation heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Features")
plt.show()

# Create age bins
bins = [ 25, 35, 45, 55, 65]
labels = [ '25-34', '35-44', '45-54', '55-64']
df['Age Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

# Boxplot using Age Group
plt.figure(figsize=(10, 5))
sns.boxplot(x="Age Group", y="Sleep Duration", data=df, palette="coolwarm")
plt.title("Sleep Duration Across Age Groups")
plt.xlabel("Age Group")
plt.ylabel("Normalized Sleep Duration")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()


# Group by Stress Level and calculate mean Physical Activity
stress_pa = df.groupby('Stress Level')['Physical Activity Level'].mean().reset_index()

plt.figure(figsize=(8, 5))
sns.barplot(x="Stress Level", y="Physical Activity Level", data=stress_pa, palette="viridis")
plt.title("Average Physical Activity Level by Stress Level")
plt.xlabel("Stress Level (Normalized)")
plt.ylabel("Avg Physical Activity Level (Normalized)")
plt.tight_layout()
plt.show()

import numpy as np
import scipy.stats as stats

sleep_data = df["Sleep Duration"].dropna()
mean_val = sleep_data.mean()
std_val = sleep_data.std()

plt.figure(figsize=(10, 5))
sns.histplot(sleep_data, bins=30, kde=True, stat="density", color="skyblue", label="Sleep Duration Distribution")

# Gaussian curve
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mean_val, std_val)
plt.plot(x, p, "r--", label="Gaussian Curve", linewidth=2)

# Mean and ±1 SD
plt.axvline(mean_val, color="green", linestyle="--", label=f"Mean: {mean_val:.2f}")
plt.axvline(mean_val + std_val, color="orange", linestyle="--", label=f"+1 SD: {mean_val + std_val:.2f}")
plt.axvline(mean_val - std_val, color="orange", linestyle="--", label=f"-1 SD: {mean_val - std_val:.2f}")

plt.title("Sleep Duration Distribution with Mean & Std Deviation")
plt.xlabel("Normalized Sleep Duration")
plt.ylabel("Density")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Group by Stress Level and calculate mean Sleep Duration
stress_sleep = df.groupby('Stress Level')['Sleep Duration'].mean().reset_index()

plt.figure(figsize=(8, 5))
sns.barplot(x="Stress Level", y="Sleep Duration", data=stress_sleep, palette="mako")
plt.title("Average Sleep Duration by Stress Level")
plt.xlabel("Stress Level (Normalized)")
plt.ylabel("Avg Sleep Duration (Normalized)")
plt.tight_layout()
plt.show()


# Count of each sleep disorder
sleep_disorder_counts = df['Sleep Disorder'].value_counts()

# Custom labels for the legend
legend_labels = { -1: "No Disorder", 0: "Insomnia", 1: "Sleep Apnea" }

# Pie chart
plt.figure(figsize=(6, 6))
colors = sns.color_palette('pastel')[0:len(sleep_disorder_counts)]
wedges, texts, autotexts = plt.pie(
    sleep_disorder_counts, labels=[legend_labels[label] for label in sleep_disorder_counts.index],
    colors=colors, autopct='%1.1f%%', startangle=140, wedgeprops={'edgecolor': 'black'}
)

# Add a legend
plt.legend(wedges, [legend_labels[label] for label in sleep_disorder_counts.index],
           title="Sleep Disorder Types", loc="best")

plt.title("Sleep Disorder Distribution")
plt.tight_layout()
plt.show()


# Drop all non-numeric columns for correlation
numeric_df = df.select_dtypes(include='number')

# Now this will work
correlations = numeric_df.corr()['Sleep Duration'].drop('Sleep Duration').sort_values()
plt.figure(figsize=(10, 5))
sns.barplot(x=correlations.values, y=correlations.index)
plt.title("Feature Correlation with Sleep Duration")
plt.xlabel("Correlation Coefficient")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

# Fix: FutureWarning - Remove palette where hue is not used
# Fix: Barplot - Stress Level vs Physical Activity Level
stress_pa = df.groupby('Stress Level')['Physical Activity Level'].mean().reset_index()
plt.figure(figsize=(8, 5))
sns.barplot(x="Stress Level", y="Physical Activity Level", data=stress_pa)
plt.title("Average Physical Activity Level by Stress Level")
plt.xlabel("Stress Level (Normalized)")
plt.ylabel("Avg Physical Activity Level (Normalized)")
plt.tight_layout()
plt.show()

# Fix: Barplot - Stress Level vs Sleep Duration
stress_sleep = df.groupby('Stress Level')['Sleep Duration'].mean().reset_index()
plt.figure(figsize=(8, 5))
sns.barplot(x="Stress Level", y="Sleep Duration", data=stress_sleep)
plt.title("Average Sleep Duration by Stress Level")
plt.xlabel("Stress Level (Normalized)")
plt.ylabel("Avg Sleep Duration (Normalized)")
plt.tight_layout()
plt.show()

# Fix: Correlation Plot (drop non-numeric)
numeric_df = df.select_dtypes(include='number')
correlations = numeric_df.corr()['Sleep Duration'].drop('Sleep Duration').sort_values()
plt.figure(figsize=(10, 5))
sns.barplot(x=correlations.values, y=correlations.index)
plt.title("Feature Correlation with Sleep Duration")
plt.xlabel("Correlation Coefficient")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

# Normalize numerical features using Min-Max Scaling
scaler = MinMaxScaler()
num_cols = ['Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 'Stress Level',
            'Heart Rate', 'Daily Steps', 'Systolic BP', 'Diastolic BP']
df[num_cols] = scaler.fit_transform(df[num_cols])
#df.to_csv("preprocessed_sleep_data.csv", index=False)
print(df)  # Prints the full DataFrame


