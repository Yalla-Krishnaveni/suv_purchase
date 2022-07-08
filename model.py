import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

df = pd.read_csv("SUV_Purchase.csv")

df = df.drop(['User ID', 'Gender'], axis=1)


X = np.array(df[['Age', 'EstimatedSalary']])
Y = np.array(df[['Purchased']])


sc = StandardScaler()
X = sc.fit_transform(X)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


model = LogisticRegression()
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)
print(y_pred)

pickle.dump(model, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))
print('success')
# Execute file only once and create the pkl file
