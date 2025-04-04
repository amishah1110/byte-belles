#Team Name: BYTE BELLES

##Problem Statement: PS-2

**Problem statement summary**
Social media is an integral part of daily life, influencing mental and physical health. This project analyzes user behavior across platforms to predict social media usage and assess its impact on well-being. The goal is to provide insights that help policymakers and individuals make informed decisions about digital consumption.

**Dataset Overview**
Two datasets were used:
1. Social Media Usage Dataset: Tracks user interactions such as time spent, posts, likes, messages, and emotions.
2. Sleep Dataset: Contains sleep patterns that may correlate with social media usage.
Both datasets contain numerical and categorical data requiring extensive preprocessing.

**Understanding and Sight of Data**
1. The dataset tracks social media usage, including time spent, posts, likes, and messages.
2. It contains numerical (e.g., usage time) and categorical (e.g., age, gender, platform, emotions) data.
3. The goal is to predict the preferred social media platform for each user.
4. Age is provided as a number and categorized for better analysis.
5. Emotions potentially influence user engagement.
6. An engagement score was created by combining various usage statistics.
7. Data is in Excel (.xlsm) format, requiring handling of multiple sheets.
8. The dataset includes behavioral (e.g., time spent) and demographic (e.g., age, gender) information.
9. Data cleaning was necessary due to missing or incorrect values.
10. Machine learning models like Random Forest and XGBoost are used for prediction.

**Data Cleaning and Transformation**
1. Skewed data check: Used standard deviation
2. Duplicity check: Not needed (no duplicates)
3. Null Check: Removed null rows
4. Data type check: Used df.dtypes and converted appropriately
5. Range check: Verified for each feature
6. Feature importance check: Used RandomForest.feature_importances_
7. Sample imputer used: Median for numerical, Mode for categorical
8. Feature engineering used: Created Age_Group and Engagement_Score
9. Standard deviation check: Checked numerical columns for outliers
10. Variance check: Used .var() to check variance


**Feature Engineering Details**
1. Age_Group: Users grouped into predefined age brackets.
2. Engagement_Score: A composite metric derived from post frequency, likes, messages, and time spent. (weighted sum of top 5 features. These features were extracted from feature scores using the best classification model)
3. Sentiment Analysis: Emotion labels mapped to numerical values to assess correlation with engagement.

![Feature importance in sleep dataset](https://github.com/user-attachments/assets/7ec11b54-ad41-4444-b520-d617c0e6e953)
![WhatsApp Image 2025-04-04 at 18 44 04_d03c5b13](https://github.com/user-attachments/assets/37504fd9-a65c-4411-bafc-c896519e0741)


**Model Selection**
Selecting the best predictive model was a crucial step. We employed the TPOT AutoML library, which automates the search for optimal machine learning pipelines. The following models were tested:
  1. Random Forest: Strong baseline model with feature importance insights.
  2. Gradient Boosting: Improved accuracy by handling complex patterns.
  3. Logistic Regression: Simpler model, useful for interpretability.
  4. Support Vector Machine (SVM): Effective for high-dimensional data.
  5. K-Nearest Neighbors (KNN): Simple distance-based classification.
  6. Naïve Bayes: Works well with categorical data but assumes independence.

**Model Performance Comparison**
![model selection](https://github.com/user-attachments/assets/aed617e0-131b-474f-ae7e-e87399245bf5)
Gradient Boosting was chosen due to good accuracy in train and test datasets


**Dashboard Images**
![WhatsApp Image 2025-04-04 at 18 57 50_25a8a434](https://github.com/user-attachments/assets/55e4b6c3-11bf-4b56-b96b-5d2b32f29ac1)
![WhatsApp Image 2025-04-04 at 18 58 06_eaaf858c](https://github.com/user-attachments/assets/01781c2e-d216-4eaa-b51f-da64b1e6c154)
![WhatsApp Image 2025-04-04 at 18 58 37_2547fe1e](https://github.com/user-attachments/assets/2f0337f6-6410-4c38-be0b-22fdd7187d87)
![WhatsApp Image 2025-04-04 at 18 58 56_72711ea3](https://github.com/user-attachments/assets/c33267dd-e089-4f73-9ef9-378cd812c016)
![WhatsApp Image 2025-04-04 at 18 59 14_811e1507](https://github.com/user-attachments/assets/13c6ed0d-b7fc-4ab6-a8a0-bb44ba57a978)









