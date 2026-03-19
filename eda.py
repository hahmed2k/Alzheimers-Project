import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('data/alzheimers_disease_data.csv')
df = df.drop(['PatientID', 'DoctorInCharge'], axis=1)

# Diagnosis Distribution
sns.countplot(x='Diagnosis', data=df, palette='viridis')
plt.title('Diagnosis Distribution')
plt.show()

# Age vs. Diagnosis
sns.boxplot(x='Diagnosis', y='Age', data=df, palette='viridis')
plt.title('Age vs. Diagnosis')
plt.show()

# BMI vs. Diagnosis
sns.boxplot(x='Diagnosis', y='BMI', data=df, palette='viridis')
plt.title('BMI vs. Diagnosis')
plt.show()

# Binary Feature Association
binary_features = [
    'Smoking', 'FamilyHistoryAlzheimers', 'CardiovascularDisease',
    'Diabetes', 'Depression', 'HeadInjury', 'Hypertension',
    'MemoryComplaints', 'BehavioralProblems',
    'Confusion', 'Disorientation', 'PersonalityChanges',
    'DifficultyCompletingTasks', 'Forgetfulness'
]

for feature in binary_features:
    print(f"\n=== {feature} ===")
    print(pd.crosstab(df[feature], df['Diagnosis'], normalize='index'))

# Correlation Heatmap
num_features = df.select_dtypes(include=[np.number])
corr_matrix = num_features.corr()
plt.figure(figsize=(20, 18))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# Feature Importance
rf = RandomForestClassifier(random_state=42)
rf.fit(num_features.drop('Diagnosis', axis=1), df['Diagnosis'])
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1][:15]
plt.bar(range(15), importances[indices])
plt.xticks(range(15), [num_features.columns[i] for i in indices], rotation=90)
plt.title('Top 15 Feature Importances')
plt.show()
