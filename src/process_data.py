import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

current_path = os.getcwd()
parent_folder = os.path.dirname(current_path)
data_file_path = os.path.join(parent_folder, "data", "Student_performance_data.csv")

df = pd.read_csv(str(data_file_path))

df['Attendance'] = 1 - (df['Absences'] / 30) #basically just the opposite of abscences data shows a max of 30 absences so divide by 30

df['Activity'] = df[['Extracurricular', 'Music', 'Sports', 'Volunteering']].sum(axis=1) #all non academic activities, probably if students are more involved they might do better academically

df['StudyTimeNorm'] = df['StudyTimeWeekly'] / 20 #study time massaged to be between 0 and 1 (data shows a max of 20 hours per week so divide by 20
#here we define the ratios of used features to create the new feature
df['Engagement'] = (
    df['Attendance'] * 0.4 +
    df['Activity'] * 0.3 +
    df['StudyTimeNorm'] * 0.3
)

df_cleaned = df.drop(columns=[
    'Age', 'Gender', 'Ethnicity', 'ParentalEducation',
    'Extracurricular', 'Music', 'Volunteering',
    'Sports', 'StudentID', 'GPA', 'StudyTimeNorm', 'Activity', 'Attendance'
])

def convert_to_risk(x):
    if x >= 3:
        return 1
    else:
        return 0

df_cleaned['AtRisk'] = df_cleaned['GradeClass'].apply(convert_to_risk)

def remove_outliers(df, cols):  #function to remove outliers
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col]<= upper)]

    return df


df_cleaned = remove_outliers(df_cleaned, ['Absences', 'StudyTimeWeekly', 'ParentalSupport', 'Engagement'])  #calls the function made above

X = df_cleaned.drop(columns=['GradeClass', 'AtRisk'])
y = df_cleaned['AtRisk'] #AtRisk = D or F

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Get the original indices
original_indices = np.arange(len(X_scaled))
# Split indices along with the data
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_scaled, y, original_indices, test_size=0.2, random_state=95
)

# Convert to Series and save
train_index = pd.Series(idx_train)
test_index = pd.Series(idx_test)

train_index_path = os.path.join(parent_folder, "data", "train_index.csv")
test_index_path = os.path.join(parent_folder, "data", "test_index.csv")

train_index.to_csv(train_index_path, index=False, header=False)
test_index.to_csv(test_index_path, index=False, header=False)

print("Part 1: Train and test indices saved to CSV.")

rf = RandomForestClassifier(n_estimators=100, random_state=101)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest Report:\n", classification_report(y_test, y_pred_rf))

joblib.dump(rf, '../artifacts/model.pkl')         # Save trained model and scaler
joblib.dump(scaler, '../artifacts/scaler.pkl') 