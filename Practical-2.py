import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

data = pd.read_csv("students_data.csv")
print("Original Dataset:\n", data)

print("\n--- Statistical Measures ---")

for col in ['StudyHours', 'Attendance', 'Score']:
    print(f"\nColumn: {col}")
    print("Mean:", data[col].mean())
    print("Median:", data[col].median())
    print("Mode:", data[col].mode()[0])

    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1

    print("Q1:", Q1)
    print("Q3:", Q3)
    print("IQR:", IQR)

data['StudyHours'].fillna(data['StudyHours'].mean(), inplace=True)

data['Score'].fillna(data['Score'].median(), inplace=True)

data['Attendance'].fillna(data['Attendance'].mode()[0], inplace=True)

print("\nAfter Handling Missing Values:\n", data)

le = LabelEncoder()
data['Gender_Label'] = le.fit_transform(data['Gender'])

print("\nLabel Encoded Gender:\n", data[['Gender', 'Gender_Label']])

one_hot = pd.get_dummies(data['Department'], prefix='Dept')
data = pd.concat([data, one_hot], axis=1)

print("\nOne-Hot Encoded Department:\n", one_hot)

dummy_vars = pd.get_dummies(data['Department'], drop_first=True)
data = pd.concat([data, dummy_vars], axis=1)

scaler = MinMaxScaler()
data[['StudyHours', 'Attendance', 'Score']] = scaler.fit_transform(
    data[['StudyHours', 'Attendance', 'Score']]
)

print("\nAfter Normalization:\n", data[['StudyHours', 'Attendance', 'Score']])