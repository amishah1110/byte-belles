import pandas as pd
import os
import time
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


print("Script execution started...", flush=True)

try:

    dataset_folder = "C:\\Users\\amisa\\PycharmProjects\\PythonProject1\\data"
    print(f"Loading data from: {dataset_folder}", flush=True)

    # Load datasets - handle FileNotFoundError more gracefully
    try:
        train_df = pd.read_csv(os.path.join(dataset_folder, "processed_train_data.csv"))
        test_df = pd.read_excel(os.path.join(dataset_folder, "Copy of Social Media Usage - Test.xlsm"))
        val_df = pd.read_excel(os.path.join(dataset_folder, "Copy of Social Media Usage - Val.xlsm"))
    except FileNotFoundError:
        print(f"Error: Dataset files not found in {dataset_folder}")
        print("Please ensure the following files exist in the data directory:")
        print("- Social Media Usage - Train.xlsm")
        print("- Social Media Usage - Test.xlsm")
        print("- Social Media Usage - Val.xlsm")
        exit(1)

    # Print dataset shapes
    print(f"Train data shape: {train_df.shape}", flush=True)
    print(f"Test data shape: {test_df.shape}", flush=True)
    print(f"Validation data shape: {val_df.shape}", flush=True)

    # Inspect the first few rows of each dataset to understand structure
    print("\nInspecting Train data (first 3 rows):")
    print(train_df.head(3).T)  # Transpose to see all columns

    # Define numerical and categorical columns based on data inspection
    num_cols = ["Daily_Usage_Time (minutes)", "Posts_Per_Day", "Likes_Received_Per_Day",
                "Comments_Received_Per_Day", "Messages_Sent_Per_Day"]

    cat_features = ["Gender", "Dominant_Emotion", "Age_Group"]

    # Check which numerical columns exist in each dataset
    print("\nChecking numerical columns:")
    for col in num_cols:
        print(f"{col}: Train: {col in train_df.columns}, Test: {col in test_df.columns}, Val: {col in val_df.columns}")


    # Preprocessing function with better handling of column differences
    def preprocess_data(df, target_encoder=None, scaler=None, fit=True):
        print(f"Preprocessing dataset with shape {df.shape}...", flush=True)
        df = df.copy()

        # Ensure platform column exists and convert to string
        if "Platform" in df.columns:
            df["Platform"] = df["Platform"].astype(str)
            # Check platform values
            print("Platform values:", df["Platform"].unique())

        # Convert numerical columns to float (if they exist)
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Fill missing values with median
                if fit:
                    col_median = df[col].median()
                    print(f"Median for {col}: {col_median}")
                    df[col] = df[col].fillna(col_median)
                else:
                    # For test/validation/user input, use training medians
                    if hasattr(preprocess_data, 'medians') and col in preprocess_data.medians:
                        df[col] = df[col].fillna(preprocess_data.medians[col])
                    else:
                        # Fallback if no stored medians
                        df[col] = df[col].fillna(df[col].median())

        # Store medians for future use (only during fit)
        if fit:
            preprocess_data.medians = {col: df[col].median() for col in num_cols if col in df.columns}

        # Handle categorical features
        if "Gender" in df.columns:
            df["Gender"] = df["Gender"].astype(str)
            df["Gender"] = df["Gender"].fillna("Unknown")

        if "Dominant_Emotion" in df.columns:
            df["Dominant_Emotion"] = df["Dominant_Emotion"].astype(str)
            df["Dominant_Emotion"] = df["Dominant_Emotion"].fillna("Unknown")

        # Convert Age to Age Groups if Age exists
        if "Age" in df.columns:
            df["Age"] = df["Age"].astype(str)

            def age_grouping(age):
                try:
                    age = int(float(age))
                    if age < 18:
                        return "Under 18"
                    elif age < 25:
                        return "18-24"
                    elif age < 35:
                        return "25-34"
                    elif age < 45:
                        return "35-44"
                    elif age < 60:
                        return "45-59"
                    else:
                        return "60+"
                except (ValueError, TypeError):
                    return "Unknown"

            df["Age_Group"] = df["Age"].apply(age_grouping)
        elif "Age_Group" not in df.columns:
            # Create Age_Group column if neither Age nor Age_Group exists
            df["Age_Group"] = "Unknown"

        # Scale numerical features
        existing_num_cols = [col for col in num_cols if col in df.columns]
        if existing_num_cols:
            if fit:
                print("Fitting scaler on numerical variables...", flush=True)
                scaler = StandardScaler()
                scaled_values = scaler.fit_transform(df[existing_num_cols])
                df[existing_num_cols] = scaled_values
            else:
                print("Transforming numerical variables with existing scaler...", flush=True)
                # Handle missing columns in test/validation/user input
                for col in scaler.feature_names_in_:
                    if col not in df.columns:
                        df[col] = 0  # Add missing column with default value

                # Ensure columns are in the correct order for the scaler
                scale_cols = [col for col in scaler.feature_names_in_ if col in df.columns]
                if len(scale_cols) > 0:
                    df[scale_cols] = scaler.transform(df[scale_cols])

        # Handle target encoding differently from feature encoding
        if "Platform" in df.columns and target_encoder is not None:
            if fit:
                print("Fitting encoder on target variable...")
                df["Platform"] = target_encoder.fit_transform(df["Platform"])
                # Store the original mapping for reference
                preprocess_data.platform_mapping = dict(
                    zip(target_encoder.classes_, target_encoder.transform(target_encoder.classes_)))
                print("Platform mapping:", preprocess_data.platform_mapping)
            else:
                print("Transforming target variable with existing encoder...")
                # Transform platform values, handle unknown values
                try:
                    df["Platform"] = target_encoder.transform(df["Platform"])
                except ValueError:
                    # For unknown values, assign a default class (most common class)
                    print("Warning: Unknown platform values in dataset. Setting to default class.")
                    most_common_idx = 0  # Default to first class if we can't determine
                    if hasattr(preprocess_data, 'most_common_platform'):
                        most_common_idx = preprocess_data.most_common_platform
                    df["Platform"] = most_common_idx

        # One-hot encode categorical variables
        # Store category values during fit for consistent encoding later
        if fit:
            preprocess_data.categories = {}

        for feature in ["Gender", "Dominant_Emotion", "Age_Group"]:
            if feature in df.columns:
                if fit:
                    # Store unique categories for future use
                    preprocess_data.categories[feature] = df[feature].unique()

                # Create dummy variables
                dummies = pd.get_dummies(df[feature], prefix=feature, drop_first=False)
                df = pd.concat([df, dummies], axis=1)

                # Drop original categorical column
                df = df.drop(feature, axis=1)
            else:
                # If the categorical column doesn't exist but we're in transform mode
                if not fit and hasattr(preprocess_data, 'categories') and feature in preprocess_data.categories:
                    # Add dummy columns with 0s for each possible category from training
                    for category in preprocess_data.categories[feature]:
                        col_name = f"{feature}_{category}"
                        df[col_name] = 0

        # For training data, determine most common platform
        if fit and "Platform" in df.columns:
            platform_counts = df["Platform"].value_counts()
            preprocess_data.most_common_platform = platform_counts.idxmax()
            print(f"Most common platform (encoded): {preprocess_data.most_common_platform}")

        print(f"Preprocessing complete. Output shape: {df.shape}", flush=True)
        return df, target_encoder, scaler


    # Create label encoder for the target variable
    target_encoder = LabelEncoder()

    # Show platforms before encoding
    print("\nPlatform values before encoding:")
    print("Train:", train_df["Platform"].unique())

    # Preprocess datasets
    processed_train_df, target_encoder, scaler = preprocess_data(train_df, target_encoder, None, fit=True)

    # Show platforms after encoding in training data
    print("\nPlatform values after encoding in train data:", processed_train_df["Platform"].unique())

    processed_test_df, _, _ = preprocess_data(test_df, target_encoder, scaler, fit=False)
    processed_val_df, _, _ = preprocess_data(val_df, target_encoder, scaler, fit=False)

    # Get all columns from the training dataset to use as features
    # This approach ensures consistency when we get new data later
    all_columns = processed_train_df.columns.tolist()
    feature_cols = [col for col in all_columns if col != "Platform"]

    # Store feature columns for prediction
    preprocess_data.feature_cols = feature_cols

    # Define features and target
    X_train = processed_train_df[feature_cols]
    y_train = processed_train_df["Platform"]

    # Ensure test and validation dataframes have all needed columns
    for col in feature_cols:
        if col not in processed_test_df.columns:
            processed_test_df[col] = 0
        if col not in processed_val_df.columns:
            processed_val_df[col] = 0

    X_test = processed_test_df[feature_cols]
    y_test = processed_test_df["Platform"]
    X_val = processed_val_df[feature_cols]
    y_val = processed_val_df["Platform"]

    print(f"\nFeature shapes - Train: {X_train.shape}, Test: {X_test.shape}, Val: {X_val.shape}", flush=True)

    # Define classifiers to compare
    classifiers = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=2000, solver='liblinear', random_state=42),
        "SVM": SVC(kernel='rbf', probability=True, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB()
    }


    # Function to evaluate model and return metrics
    def evaluate_model(model, X_train, y_train, X_test, y_test):
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Calculate metrics
        metrics = {
            "Train Accuracy": accuracy_score(y_train, y_train_pred),
            "Test Accuracy": accuracy_score(y_test, y_test_pred),
            "Test Precision": precision_score(y_test, y_test_pred, average='weighted', zero_division=0),
            "Test Recall": recall_score(y_test, y_test_pred, average='weighted', zero_division=0),
            "Test F1": f1_score(y_test, y_test_pred, average='weighted', zero_division=0),
            "Training Time": train_time
        }

        return metrics


    # Store results
    results = []

    # Train and evaluate each classifier
    print("\nTraining and evaluating classifiers:")
    for name, clf in classifiers.items():
        print(f"\nTraining {name}...", flush=True)

        try:
            metrics = evaluate_model(clf, X_train, y_train, X_test, y_test)
            metrics["Classifier"] = name
            results.append(metrics)
            print(f"{name} completed successfully!", flush=True)
        except Exception as e:
            print(f"Error training {name}: {str(e)}", flush=True)

    # Create DataFrame for displaying results
    results_df = pd.DataFrame(results)

    # Reorder columns
    col_order = ["Classifier", "Train Accuracy", "Test Accuracy", "Test Precision", "Test Recall", "Test F1",
                 "Training Time"]
    results_df = results_df[col_order]

    # Format numeric columns to 4 decimal places
    for col in results_df.columns:
        if col != "Classifier" and col != "Training Time":
            results_df[col] = results_df[col].map(lambda x: f"{x:.4f}")

    results_df["Training Time"] = results_df["Training Time"].map(lambda x: f"{x:.2f}")

    # Display results
    print("\n=== MODEL COMPARISON ===", flush=True)
    try:
        print(results_df.to_string(index=False), flush=True)
    except:
        print(results_df, flush=True)

    # Find best model based on test accuracy
    best_idx = np.argmax([float(r["Test Accuracy"]) for r in results])
    best_classifier = results[best_idx]["Classifier"]
    print(f"\nBest classifier based on test accuracy: {best_classifier}", flush=True)

    # Retrieve the best trained model
    best_model = classifiers[best_classifier]

    # Save the mapping of encoded values to platform names
    platform_names = target_encoder.classes_


    # Improved user input function
    def predict_user_input():
        user_data = {}

        print("\n===== Social Media Platform Prediction =====")
        print("Please enter your details to predict which platform you're most likely to use:\n")

        # Collect numerical inputs with validation
        for col in num_cols:
            while True:
                try:
                    value = input(f"Enter {col.replace('_', ' ')}: ")
                    user_data[col] = float(value)
                    break
                except ValueError:
                    print("Please enter a valid number")

        # Collect categorical inputs
        # Gender
        print("\nSelect your gender:")
        print("1. Male")
        print("2. Female")
        print("3. Non-binary")
        print("4. Other")
        gender_choice = input("Enter number (1-4): ")
        gender_map = {"1": "Male", "2": "Female", "3": "Non-binary", "4": "Other"}
        user_data["Gender"] = gender_map.get(gender_choice, "Unknown")

        # Age group
        print("\nSelect your age group:")
        print("1. Under 18")
        print("2. 18-24")
        print("3. 25-34")
        print("4. 35-44")
        print("5. 45-59")
        print("6. 60+")
        age_choice = input("Enter number (1-6): ")
        age_map = {"1": "Under 18", "2": "18-24", "3": "25-34",
                   "4": "35-44", "5": "45-59", "6": "60+"}
        user_data["Age_Group"] = age_map.get(age_choice, "Unknown")

        # Dominant emotion
        print("\nWhat's your dominant emotion when using social media?")
        print("1. Joy")
        print("2. Excitement")
        print("3. Interest")
        print("4. Boredom")
        print("5. Anxiety")
        print("6. Anger")
        print("7. Sadness")
        print("8. Other")
        emotion_choice = input("Enter number (1-8): ")
        emotion_map = {"1": "Joy", "2": "Excitement", "3": "Interest", "4": "Boredom",
                       "5": "Anxiety", "6": "Anger", "7": "Sadness", "8": "Other"}
        user_data["Dominant_Emotion"] = emotion_map.get(emotion_choice, "Unknown")

        # Create DataFrame with user input
        user_df = pd.DataFrame([user_data])

        # Process the user input
        user_processed, _, _ = preprocess_data(user_df, target_encoder, scaler, fit=False)

        # Ensure all required features are present
        for col in feature_cols:
            if col not in user_processed.columns:
                user_processed[col] = 0

        # Get only the needed columns in the right order
        user_features = user_processed[feature_cols]

        # Make prediction
        prediction = best_model.predict(user_features)[0]

        # Get probabilities for all platforms
        probabilities = best_model.predict_proba(user_features)[0]

        # Map encoded prediction back to platform name
        platform_pred = target_encoder.inverse_transform([prediction])[0]

        # Display results
        print("\n===== Results =====")
        print(f"Predicted Platform: {platform_pred}")

        # Display top 3 most likely platforms with probabilities
        print("\nTop platform probabilities:")
        platform_probs = [(platform_names[i], prob * 100) for i, prob in enumerate(probabilities)]
        platform_probs.sort(key=lambda x: x[1], reverse=True)

        for platform, prob in platform_probs[:3]:
            print(f"{platform}: {prob:.2f}%")


    # Call function to get user input and predict
    predict_user_input()

except Exception as e:
    print(f"Error occurred: {str(e)}", flush=True)
    import traceback

    print(traceback.format_exc(), flush=True)