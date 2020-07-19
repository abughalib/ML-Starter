import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("titanic_dataset.csv")

print(df.head())

# Method -> 1 Removing N.a.N
# Calculation of survial & pclass

print(df.isna().sum())

df = df.dropna()

# Removing outliers
q_low = df["Fare"].quantile(0.01)
q_hi = df["Fare"].quantile(0.99)

no_out = df[(df["Fare"] < q_hi) & (df["Fare"] > q_low)]

seat_class = no_out.iloc[:, [2]].values
seat_fair = no_out.iloc[:, [-3]].values

plt.scatter(seat_fair, seat_class)
plt.title("Fair v.s pClass")
plt.show()

no_out.hist()


# Fair vs pClass Regression

def warn(*arg, **kwrg):
    pass


import warnings

warnings.warn = warn

id = df.iloc[:, [0]].values
surviver = df.iloc[:, [1]].values

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(id, surviver)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(x_train, y_train);

y_predict = model.predict(x_test)

plt.scatter(x_train, y_train, label="Original")
plt.scatter(x_test, y_predict, label="Predicated");
plt.legend()
plt.show()

# Method -> 2
# 1. Split the data into two test and train
# 2.  Visualize the data
# 3. and Fill the Missing Value
# 4. Creating model


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# print(test.head())
# print(train.head())

# print(test.info(), train.info())

# Checking if NAN is present
print(train.isna().sum(), test.isna().sum())

# 1. Visualization of Data
# Using python Seaborn lib for visulalization
import seaborn as sns

sns.set()  # seaborn default ploting


def bar_chart(feature):
    survived = train[train['Survived'] == 1][feature].value_counts()
    # selecting particular feature <where survived=1>
    dead = train[train['Survived'] == 0][feature].value_counts()
    # selecting particular feature <where survived=0>
    df = pd.DataFrame([survived, dead])
    df.index = ['Survived', 'Dead']
    df.plot(kind='bar', stacked=True, figsize=(10, 5))
    plt.show()
    # Stacked -> to draw on top
    # figsize -> bar width


bar_chart('Sex')
bar_chart("Pclass")
bar_chart('SibSp')
bar_chart('Parch')
bar_chart('Embarked')

# Using Name
# Combining train and test dataset
train_test_data = [train, test]

# Extracting title from dataset.

for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    # Regex for extracting title

print(train['Title'].value_counts())
print(test['Title'].value_counts())

# Dumping data as Mr, Miss, Mrs, Master etc.... as 0,1,2,3

title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2,
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }

# Inserting title into dataset
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    # Map title to dictionary

train.drop('Name', axis=1, inplace=True)
test.drop('Name', axis=1, inplace=True)
train.head()
# print(train.head())
bar_chart('Title')

# Using Sex Data

sex_mapping = {'male': 1, "female": 0}
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)

bar_chart('Sex')

# Using Age Data

# Filling missing age with respect to title(Mr, Miss, etc..)
# From the data given
train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace=True)
test["Age"].fillna(test.groupby("Title")["Age"].transform("median"), inplace=True)

train.groupby("Title")["Age"].transform("median")

fg = sns.FacetGrid(train, hue="Survived", aspect=4)
#Multi-plot grid for plotting conditional relationships.

fg.map(sns.kdeplot, 'Age', shade=True)
#Fit and plot a univariate or bivariate kernel density estimate.

fg.set(xlim=(0, train['Age'].max()))
#Above line-> xlim from 0 to max age
fg.add_legend()
plt.show()
fg = sns.FacetGrid(train, hue="Survived", aspect=4)
fg.map(sns.kdeplot, 'Age', shade=True)
fg.set(xlim=(0, train['Age'].max()))
fg.add_legend()
plt.xlim(20,30)
# zoom over 20 to 30

print(train.info(), test.info())

# Age Conversion
''' Converting Numerical Age to Categorial Variable

age > 16 -> 0
age > 26 > 16 -> 1
age > 36 > 26 -> 2
age > 62 > 36 -> 3
age > 62 -> 4
'''
for dataset in train_test_data:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0,
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 26), 'Age'] = 1,
    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age'] = 2,
    dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 62), 'Age'] = 3,
    dataset.loc[ dataset['Age'] > 62, 'Age'] = 4

bar_chart('Age')

# Using Embarked

pcls1 = train[train['Pclass']==1]["Embarked"].value_counts()
pcls2 = train[train['Pclass']==2]['Embarked'].value_counts()
pcls3 = train[train['Pclass']==3]['Embarked'].value_counts()

df = pd.DataFrame([pcls1, pcls2, pcls3])
df.index = ['1st Class', '2nd Class', '3rd Class']
df.plot(kind='bar', stacked=True, figsize=(10,5))

# Filling out missing embark with S embark
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')

embarked_mapping = {"S": 0, "C": 1, "Q": 2}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)

# Using Fare

#fill missing fare with median fare for each Pclass
train["Fare"].fillna(train.groupby("Pclass")["Fare"].transform("median"), inplace=True)
test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace=True)
train.head(50)
fg = sns.FacetGrid(train, hue="Survived", aspect=4)
fg.map(sns.kdeplot, 'Fare', shade=True)
fg.set(xlim=(0, train['Fare'].max()))
fg.add_legend()
plt.show()

fg = sns.FacetGrid(train, hue="Survived", aspect=4)
fg.map(sns.kdeplot, 'Fare', shade=True)
fg.set(xlim=(0, train['Fare'].max()))
fg.add_legend()
plt.xlim(0,30) # Zooming x to 0 to 20.
plt.show()

'''
### Transforming
Fare > 17 -> 0
Fare > 17 <= 30 -> 1 
Fare > 30 <=100 -> 2
Fare > 100 -> 3
'''

for dataset in train_test_data:
    dataset.loc[ dataset['Fare'] <= 17, 'Fare'] = 0,
    dataset.loc[(dataset['Fare'] > 17) & (dataset['Fare'] <= 30), 'Fare'] = 1,
    dataset.loc[(dataset['Fare'] > 30) & (dataset['Fare'] <= 100), 'Fare'] = 2,
    dataset.loc[ dataset['Fare'] > 100, 'Fare'] = 3

# Using Cabin Data and Filling it

print(train.Cabin.value_counts())

for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].str[:1]

pcls1 = train[train['Pclass']==1]['Cabin'].value_counts()
pcls2 = train[train['Pclass']==2]['Cabin'].value_counts()
pcls3 = train[train['Pclass']==3]['Cabin'].value_counts()

df = pd.DataFrame([pcls1, pcls2, pcls3])
df.index=['1st Class', '2nd Class', '3rd Class']
df.plot(kind='bar', stacked=True, figsize=(10,5))

cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}

for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)

#fill missing fare with median fare for each Pclass
train["Cabin"].fillna(train.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
test["Cabin"].fillna(test.groupby("Pclass")["Cabin"].transform("median"), inplace=True)

# Using Family Size

train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
test["FamilySize"] = test["SibSp"] + test["Parch"] + 1

fg = sns.FacetGrid(train, hue="Survived",aspect=4)
fg.map(sns.kdeplot,'FamilySize',shade= True)
fg.set(xlim=(0, train['FamilySize'].max()))
fg.add_legend()
plt.xlim(0)

# Mapping
''' 1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2, 7: 2.4, 8: 2.8,
 9: 3.2, 10: 3.6, 11: 4
 '''
family_mapping = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4}

for dataset in train_test_data:
    dataset['FamilySize'] = dataset['FamilySize'].map(family_mapping)

#print(train.head())


features_drop = ['Ticket', 'SibSp', 'Parch']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)
train = train.drop(['PassengerId'], axis=1)


train_data = train.drop('Survived', axis=1)
target = train['Survived']

# print(train_data.shape, target.shape)

# print(train_data.head())

# Creating Models

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import numpy as np

# print(train.info())

# Cross Validation -> KFold
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

clf = KNeighborsClassifier(n_neighbors=13)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)

#Score
print(f"{round(np.mean(score)*100,2)}%")

# Decision Tree

clf = DecisionTreeClassifier()
scoring= 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
print(f"{round(np.mean(score)*100, 2)}%")

# Random Forest

clf = RandomForestClassifier(n_estimators=13)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)

print(f"{round(np.mean(score)*100, 2)}%")

# Naive Bayes
clf = GaussianNB()
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring='accuracy')
print(score)

print(f"{round(np.mean(score)*100, 2)}%")

# SVM
clf = SVC()
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring='accuracy')
print(score)

print(f"{round(np.mean(score)*100, 2)}%")

# Testing

clf = SVC()
clf.fit(train_data, target)

test_data = test.drop("PassengerId", axis=1).copy()
prediction = clf.predict(test_data)

# Output y_predict to file
sub = pd.DataFrame({
    "PassengerId": test['PassengerId'],
    "Survived": prediction
})

sub.to_csv("sub.csv", index=False)

# Reading output again
sub = pd.read_csv("sub.csv")
print(sub.head())