import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
df=pd.read_csv("data/loan_data.csv")
le=LabelEncoder()
cols=[
    'Gender', 'Married', 'Dependents',
    'Education', 'Self_Employed', 'Property_Area', 'Loan_Status'
]

for col in cols:
    df[col] = le.fit_transform(df[col])
X= df.drop(['Loan_ID', 'Loan_Status'], axis=1)
y= df['Loan_Status']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model= RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

model.fit(X_train, y_train)
joblib.dump(model, "models/loan_model.pkl")
print("Model trained and saved successfully!")