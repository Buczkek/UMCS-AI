import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

data = pd.read_csv("data_2.csv")

df = pd.DataFrame(data)

x = df.drop(['Survived'], axis=1)
y = df.iloc[:, 1]

sc = MinMaxScaler()
le = LabelEncoder()
x = x.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)

x['Age'].fillna(x['Age'].mean(), inplace=True)
x['Embarked'].fillna(x['Embarked'].mode(), inplace=True)
x['Fare'].fillna(x['Fare'].mean(), inplace=True)
x['Age'] = x['Age'].astype('int')

x = x.join(pd.get_dummies(x['Sex']))#, drop_first=True))
x = x.join(pd.get_dummies(x['Embarked']))#, drop_first=True))

del x['Embarked']
del x['Sex']

# x['Sex'] = le.fit_transform(x['Sex'])
# x['Embarked'] = le.fit_transform(x['Embarked'])

score = 0

x = sc.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)
#x_train = sc.fit_transform(x_train)
#x_test = sc.transform(x_test)
model = SVC(C=1.1)
model.fit(x_train, y_train)
score += model.score(x_test, y_test)
#print(y_train.value_counts(), y_test.value_counts())
print(model.score(x_test, y_test))
#print("Average model score:", score / 5)
