import pandas as pd

dataset = pd.read_csv("Social_Network_Ads.csv")

print(dataset.isna().sum())

print(dataset.describe())

a = list()
b = list()

for i in range(len(dataset.Purchased)):
    if dataset.iloc[i].Purchased == 1:
        a.append([dataset.iloc[i].Age, dataset.iloc[i].EstimatedSalary])
    else:
        b.append([dataset.iloc[i].Age, dataset.iloc[i].EstimatedSalary])

import numpy as np

A = np.array(a)
B = np.array(b)

import matplotlib.pyplot as plt

plt.title("Purchased or not")
plt.xlabel("Age")
plt.ylabel("Salary")
plt.scatter(A[:, 0], A[:, 1], marker='x', color='g', label='YES')
plt.scatter(B[:, 0], B[:, 1], marker='o', color='r', label='NO')
plt.legend()
plt.show()

X = dataset.iloc[:, [2, 3]].values
Y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=7)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_predict)
accuracy = (cm[0][0] + cm[1][1]) * 100 / np.sum(cm)
print(cm, accuracy)

# Accuracy Function from sklearn
from sklearn.metrics import accuracy_score

print(f"Accuracy: {accuracy_score(y_test, y_predict) * 100}%")

# 3-D Graph

x = dataset.Age
y = dataset.EstimatedSalary
z = dataset.Purchased

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x, y, z, color='r', marker='*')
ax.set_xlabel('X-Axis')
ax.set_ylabel('Y-Axis')
ax.set_zlabel('Z-Axis')

plt.show()


# Predicted 3-D Graph
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x_test[:,0], x_test[:,1], y_test, color='g', marker='x')
ax.set_xlabel('X-Axis')
ax.set_ylabel('Y-Axis')
ax.set_zlabel('Z-Axis')

plt.show()

# Relative 3-D Graph
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x_test[:,0], x_test[:,1], y_test, color='r', marker='x')
ax.plot(x_test[:,0], x_test[:,1], y_predict, color='g', marker='x')
ax.set_xlabel('X-Axis')
ax.set_ylabel('Y-Axis')
ax.set_zlabel('Z-Axis')
plt.show()

# Using Gender
dummy_data = pd.get_dummies(dataset['Gender'], drop_first=True)

df = pd.concat([dummy_data, dataset], axis=1)

x = df.iloc[:,[0,3,4]].values
y = df.iloc[:,[-1]].values

print(x,y)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.3,
                                                    random_state=7)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_predict)*100)


