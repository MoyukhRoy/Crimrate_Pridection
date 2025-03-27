import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

crime_data = pd.read_csv('/Users/user/Desktop/crime_rate/crime.csv')
crime_data.dropna(inplace=True)

label_encoder = LabelEncoder()
crime_data['TYPE'] = label_encoder.fit_transform(crime_data['TYPE'])

X = crime_data[['Latitude', 'Longitude', 'TYPE']]
y = crime_data['NEIGHBOURHOOD']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Making predictions
y_pred = clf.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")