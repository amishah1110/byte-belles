import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# File paths
train_path = "D:\\Sanika\\Datathon\\byte-belles\\Data\\MergedData\\Merged_Train_Top5_engineered.csv"
test_path = "D:\\Sanika\\Datathon\\byte-belles\\Data\\MergedData\\Merged_Test_Top5_engineered.csv"

# Load datasets
train_data = pd.read_csv(train_path).dropna()
test_data = pd.read_csv(test_path).dropna()

# Define target variable (Mental_Health_Effect)
conditions = [
    (train_data['Stress Impact Score'] > 7) & (train_data['Sleep Duration'] < 6),
    (train_data['Stress Impact Score'] <= 7) & (train_data['Sleep Duration'] >= 6)
]
labels = ['Negative', 'Positive']
train_data['Mental_Health_Effect'] = np.select(conditions, labels, default='Neutral')

conditions_test = [
    (test_data['Stress Impact Score'] > 7) & (test_data['Sleep Duration'] < 6),
    (test_data['Stress Impact Score'] <= 7) & (test_data['Sleep Duration'] >= 6)
]
test_data['Mental_Health_Effect'] = np.select(conditions_test, labels, default='Neutral')

# Encode categorical target
label_encoder = LabelEncoder()
train_data['Mental_Health_Effect'] = label_encoder.fit_transform(train_data['Mental_Health_Effect'])
test_data['Mental_Health_Effect'] = label_encoder.transform(test_data['Mental_Health_Effect'])

# Define features and target
X_train = train_data.drop(columns=['Mental_Health_Effect'])
y_train = train_data['Mental_Health_Effect']
X_test = test_data.drop(columns=['Mental_Health_Effect'])
y_test = test_data['Mental_Health_Effect']

# Convert categorical features to numerical
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

# Ensure same feature columns in train and test
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    "ExtraTrees": ExtraTreesClassifier(),
    "RandomForest": RandomForestClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "DecisionTree": DecisionTreeClassifier(),
    "LibLinear_SVC": LinearSVC(dual=False)
}

# Train and evaluate models
results = []
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')

    results.append([name, accuracy, f1, precision, recall])

# Convert results to DataFrame
results_df = pd.DataFrame(results, columns=["Model Name", "Accuracy", "F1 Score", "Precision", "Recall"])
print(results_df)