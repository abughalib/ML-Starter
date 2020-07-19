import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Position_Salary.csv")

X = df.iloc[:,1:2]
Y = df.iloc[:,-1]

x = X.to_numpy()
y = Y.to_numpy()

x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)
y_predict = model.predict(x_test)

y_test = np.array(y_test, dtype=float)

print(y_test, y_predict)

plt.scatter(x_test, y_test, marker="*", color='r')
plt.plot(x_test, y_predict)
plt.show()