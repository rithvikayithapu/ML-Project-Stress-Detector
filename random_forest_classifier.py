"""# **Machine Learning Project**"""

"""## **HELPER FUNCTIONS**"""

"""### SAVE AND LOAD MODELS"""

import pickle

def save_model(model, file_path):
  pickle.dump(model, open(file_path, 'wb'))

def load_model(file_path):
  return pickle.load(open(file_path), 'rb')

"""## **LOAD DATASET**"""

import numpy as np
import pandas as pd

df = pd.read_csv('stress_dataset4.csv')

fields = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
df_data = df[['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']]
df_target = df[['stressed']]

"""## **TEST MODELS**"""

from sklearn.model_selection import train_test_split
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(df_data.to_numpy(), df_target.to_numpy().ravel(), test_size=0.5,random_state=109)

"""### RANDOM FOREST MODEL"""

from sklearn.ensemble import RandomForestClassifier
RandomForestModel = RandomForestClassifier(n_estimators=100)

RandomForestModel.fit(X_train, y_train)
y_pred_rf = RandomForestModel.predict(X_test)

print("Random Forest Model Analysis:\n")
print("Confusion Matrix:\n", metrics.confusion_matrix(y_test, y_pred_rf))
print("Accuracy:", metrics.accuracy_score(y_test, y_pred_rf))
print("Precison:", metrics.precision_score(y_test, y_pred_rf, average="weighted"))
print("Recall:", metrics.recall_score(y_test, y_pred_rf, average="weighted"))
print("F-Score:", metrics.f1_score(y_test, y_pred_rf, average="weighted"))

save_model(RandomForestModel, "rfm.sav")

