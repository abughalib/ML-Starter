import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Math

df = pd.read_csv("titanic_dataset.csv")
df.head()

df.isna().sum()
df.Fare.max()
df = df.dropna()
df.isna().sum()

q_low = df["Fare"].quantile(0.01)
q_hi = df["Fare"].quantile(0.99)
no_out = df[(df["Fare"] < q_hi) & (df["Fare"] > q_low)]
seat_class = no_out.iloc[:, [2]].values
seat_fair = no_out.iloc[:, [-3]].values
plt.scatter(seat_fair, seat_class)
plt.title("Fair v.s pClass")
plt.show()

no_out.hist()

def warn(*arg, **kwrg):
    pass
import warnings
warnings.warn = warn
id = df.iloc[:, [0]].values
surviver = df.iloc[:,[1]].values
x_train, x_test, y_train, y_test = train_test_split(id, surviver)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train);
y_predict = model.predict(x_test)
plt.scatter(x_train, y_train, label="Original")
plt.scatter(x_test, y_predict, label="Predicated");
plt.legend()
plt.show()

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train.head()

test.head()

train.shape, test.shape

train.info(), test.info()

#Checking if NAN is present
train.isna().sum(), test.isna().sum()


#Using python Seaborn lib for visulalization
import seaborn as sns
sns.set() #seaborn default ploting

def bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    #selecting particular feature <where survived=1>
    dead = train[train['Survived']==0][feature].value_counts()
    #selecting particular feature <where survived=0>
    df = pd.DataFrame([survived, dead])
    df.index = ['Survived', 'Dead']
    df.plot(kind='bar', stacked=True, figsize=(10,5))
#Stacked -> to draw on top
#figsize -> bar width

bar_chart('Sex')

bar_chart("Pclass")

bar_chart('SibSp')

bar_chart('Parch')

bar_chart('Parch')

bar_chart('Embarked')

train.head()

#Combining train and test dataset
train_test_data = [train, test]
#Extracting title from dataset.
for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False
    #Regex for extracting title
train['Title'].value_counts()
test['Title'].value_counts()
                                               
#Dumping data as Mr, Miss, Mrs, Master etc.... as 0,1,2,3
title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2,
"Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,
"Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme":
#Inserting title into dataset
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)
#Map tile to dictionary  
                 
train.drop('Name', axis=1, inplace=True)
test.drop('Name', axis=1, inplace=True)
train.head()
                 
test.head()
                 
bar_chart('Title')
                 
sex_mapping = {'male': 1, "female": 0}
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)

bar_chart('Sex')
                 
train.head(100)

                 #Filling missing age with respect to title(Mr, Miss, etc..)
#From the data given
train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace=True
test["Age"].fillna(test.groupby("Title")["Age"].transform("median"), inplace=True
               
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
#zoom over 20 to 30
                   
train.info(), test.info()                  

for dataset in train_test_data:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0,
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 26), 'Age'] = 1,
    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age'] = 2,
    dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 62), 'Age'] = 3,
    dataset.loc[ dataset['Age'] > 62, 'Age'] = 4

   
bar_chart('Age')

pcls1 = train[train['Pclass']==1]["Embarked"].value_counts()
pcls2 = train[train['Pclass']==2]['Embarked'].value_counts()
pcls3 = train[train['Pclass']==3]['Embarked'].value_counts()
df = pd.DataFrame([pcls1, pcls2, pcls3])
df.index = ['1st Class', '2nd Class', '3rd Class']
df.plot(kind='bar', stacked=True, figsize=(10,5))

# Filling out missing embark with S embark
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
                   
train.head()
                   
embarked_mapping = {"S": 0, "C": 1, "Q": 2}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)
                   
#fill missing fare with median fare for each Pclass
train["Fare"].fillna(train.groupby("Pclass")["Fare"].transform("median"), inplace
test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace=T
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
                    
for dataset in train_test_data:
    dataset.loc[ dataset['Fare'] <= 17, 'Fare'] = 0,
    dataset.loc[(dataset['Fare'] > 17) & (dataset['Fare'] <= 30), 'Fare'] = 1,
    dataset.loc[(dataset['Fare'] > 30) & (dataset['Fare'] <= 100), 'Fare'] = 2,
    dataset.loc[ dataset['Fare'] > 100, 'Fare'] = 3
                    
#train.head()
                    
train.Cabin.value_counts()
                    
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].str[:1]
                    
pcls1 = train[train['Pclass']==1]['Cabin'].value_counts()
pcls2 = train[train['Pclass']==2]['Cabin'].value_counts()
pcls3 = train[train['Pclass']==3]['Cabin'].value_counts()
df = pd.DataFrame([pcls1, pcls2, pcls3])
df.index=['1st Class', '2nd Class', '3rd Class']
df.plot(kind='bar', stacked=True, figsize=(10,5))
                    
cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)
                 
#fill missing fare with median fare for each Pclass
train["Cabin"].fillna(train.groupby("Pclass")["Cabin"].transform("median"), inplace
test["Cabin"].fillna(test.groupby("Pclass")["Cabin"].transform("median"), inplace
                     
train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
test["FamilySize"] = test["SibSp"] + test["Parch"] + 1
                     
fg = sns.FacetGrid(train, hue="Survived",aspect=4)
fg.map(sns.kdeplot,'FamilySize',shade= True)
fg.set(xlim=(0, train['FamilySize'].max()))
fg.add_legend()
plt.xlim(0) 

family_mapping = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2, 7: 2.4, 8: 2.8, 9:
for dataset in train_test_data:
    dataset['FamilySize'] = dataset['FamilySize'].map(family_mapping)
train.head()  
                  
features_drop = ['Ticket', 'SibSp', 'Parch']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)
train = train.drop(['PassengerId'], axis=1)
                  
train_data = train.drop('Survived', axis=1)
target = train['Survived']
train_data.shape, target.shape
                  
train_data.head(10)
                  
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import numpy as np
                  
train.info()

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
                  
clf = KNeighborsClassifier(n_neighbors=13)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=sco
print(score)
                        
#Score
print(f"{round(np.mean(score)*100,2)}%")
                        
train_data.head()
                        
clf = DecisionTreeClassifier()
scoring= 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scorin
print(score)                       
                  
clf = RandomForestClassifier(n_estimators=13)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=sco
print(score)
                        
f"{round(np.mean(score)*100, 2)}%"
                        
clf = GaussianNB()
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring='ac
print(score)
                        
f"{round(np.mean(score)*100, 2)}%"                        
                        
clf = SVC()
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring='ac
print(score)
                        
f"{round(np.mean(score)*100, 2)}%"
                        
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
sub.head()
                   
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
