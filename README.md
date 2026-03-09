# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 1.Collect spam and non-spam emails and preprocess the text.
2.Convert email text into numerical form using TF-IDF / Bag of Words.
3.Train the Support Vector Machine using training data.
4.Test the model and calculate accuracy.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: SHREYAS M
RegisterNumber: 25013237 
*/
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = pd.read_csv(r"C:/Users/acer/Downloads/spam.csv", encoding="latin-1")

data = data[['v1', 'v2']]
data.columns = ['label', 'message']

data['label'] = data['label'].map({'ham': 0, 'spam': 1})

X = data['message']
y = data['label']

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = SVC(kernel='linear')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n")
print(cm) 
```

## Output:
<img width="831" height="495" alt="Screenshot 2026-03-09 085254" src="https://github.com/user-attachments/assets/291a2f02-8198-4407-8349-09a3ec310e97" />



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
