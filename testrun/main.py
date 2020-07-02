import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("50_Startups.csv")

dummy_data = pd.get_dummies(df.State, drop_first=True)

df = pd.concat([df, dummy_data], axis=1)

df.pop('State')

x = df.iloc[:,[0,1,2,4,5]].values
y = df.iloc[:,[3]].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)

model = LinearRegression()
model.fit(x_train, y_train)
y_predict = model.predict(x_test)

#plt.scatter(x, y, marker='*', color='r')
plt.plot(y, y_predict, marker='x', color='g')
plt.show()

