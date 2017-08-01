'''
Class 5 Data Cleaning and Feature Engineering; Missing values
https://www.kaggle.com/helgejo/an-interactive-data-science-tutorial/data/notebook
'''
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
import numpy as np
import pandas as pd

path = 'data/'

train = pd.read_csv(path+'train.csv')
test = pd.read_csv(path+'test.csv')

print '%s instances in training set' % len(train)
print '%s features in training set' % list(train)

print '\n%s instances in test set' % len(test)
print '%s features in test set' % list(test) 

print 'feature missing in test set %s' % list( set(list(train)) - set(list(test)) )
# Submission - {'PassengerId' : [], 'Survived' : 0/1}

titanic = train.append(test, ignore_index = True)
print '\n%s instances in complete set' % len(titanic)


# Dependent variable
train_target = train['Survived']
train_features = train.drop(['Survived'], axis=1)


# print list(train)
# ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

# Missing values
# print train.isnull().sum()
# print test.isnull().sum()

# null accuracy
# print train_target.value_counts()
# print train_target.value_counts(normalize=True)

# print train.describe()

#visualization

import matplotlib.pyplot as plt
# train = titanic
labels = list(set(train))
# plt.matshow(train.corr(), labels=labels)

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(train.corr())
for (i, j), z in np.ndenumerate(train.corr()):
    ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')

fig.colorbar(cax)

ax.set_xticklabels(labels)
ax.set_yticklabels(labels)

# plt.show()

# Passenger Class
pclass = train.groupby(['Pclass', 'Survived'])['Pclass'].count().unstack('Survived').fillna(0)

# pclass = pclass.div(pclass.sum(axis=1), axis=0)
pclass[[0,1]].plot(kind='bar', stacked=True)
# print '\n', pclass
# plt.show()

# Embarked
emb = train.groupby(['Embarked', 'Survived'])['Embarked'].count().unstack('Survived').fillna(0)

emb = emb.div(emb.sum(axis=1), axis=0)
emb[[0,1]].plot(kind='bar', stacked=True)
# print '\n', pclass
# plt.show()


# Categorical values
# print '\n', titanic.info()
# print titanic.head()

# Cabin, Sex, Embarked
# cabin = pd.get_dummies(titanic['Cabin'], prefix='cabin' )
# print cabin.head()

# embarked = pd.get_dummies(titanic['Embarked'], prefix='embarked' )
# print embarked.head()

# Replace columns
# titanic = pd.concat([titanic,embarked],axis=1)
# print titanic.head()
# 
# titanic.drop(['Embarked'], axis=1, inplace = True)
# print titanic.head()

# sex
# titanic['female'] = pd.Series(np.where(titanic.Sex == 'female' , 1 , 0 ), name = 'Sex' )
# titanic.drop(['Sex'], axis=1, inplace = True)
# print titanic.head()

# Imputation
# titanic['age'] = titanic['Age'].fillna(titanic['Age'].mean())
# titanic['fare'] = titanic['Fare'].fillna(titanic['Fare'].mean())
# print titanic.head()
# titanic.drop(['age','fare'], axis=1, inplace = True)

# Text columns - Name, Ticket

# Title from name
# a map of more aggregated titles
# title_map = {'Capt' : 'Officer', 'Col' : 'Officer', 'Major' : 'Officer','Dr' : 'Officer', 'Rev' : 'Officer',
#                     'Jonkheer' : 'Royalty', 'Don' : 'Royalty', 'Sir' : 'Royalty', 'the Countess' : 'Royalty', 
#                     'Dona' : 'Royalty','Lady' : 'Royalty',
#                     'Mme' : 'Mrs', 'Mrs' : 'Mrs', 'Ms' : 'Mrs',
#                     'Mlle' : 'Miss', 'Miss' : 'Miss',
#                     'Mr' :  'Mr',
#                     'Master' : 'Master'
#                     }
# titanic['title'] = titanic['Name'].map( lambda name: name.split( ',' )[1].split( '.' )[0].strip() )
# titanic['title'] = titanic['title'].map(title_map)
# title = pd.get_dummies(titanic['title'])
# 
# titanic = pd.concat([titanic,title] , axis = 1 )
# titanic.drop(['title'],axis=1)
# print titanic.head()

# TODO try plotting the chances of survival on title

# cabin
# 
# # replacing missing cabins with U (for Uknown)
# titanic['Cabin'] = titanic.Cabin.fillna('U')
# 
# # mapping each Cabin value with the cabin letter
# titanic['Cabin'] = titanic['Cabin'].map(lambda c : c[0])
# 
# # dummy encoding ...
# cabin = pd.get_dummies( cabin['Cabin'], prefix = 'Cabin' )
# 
# cabin.head()
# titanic = pd.concat([titanic,cabin], axis = 1 )
# titanic.drop(['Cabin'],axis=1)
# print titanic.head()


# Check for correlation and remove correlated variables


# datasets for Cross Validation
train = titanic[:891]
train_target = train['Survived']
train_features = train.drop(['Survived'], axis=1)

test = titanic[891:]
# print len(test)
from sklearn.cross_validation import train_test_split, StratifiedKFold
x_train, x_test, y_train, y_test = train_test_split(train_features, train_target, test_size=0.3, random_state=0)

model = LogisticRegression()
model.fit(x_train, y_train)
print (model.score(x_train, y_train), model.score(x_test , y_test))
rfecv = RFECV( estimator = model , step = 1 , cv = StratifiedKFold(y_train , 2 ) , scoring = 'accuracy' )
rfecv.fit(x_train, y_train)

     

test_Y = model.predict(x_test)
passenger_id = titanic[891:].PassengerId
test = pd.DataFrame({'PassengerId': passenger_id , 'Survived': test_Y } )
test.shape
test.head()
test.to_csv( 'titanic_pred.csv' , index = False )


# save models and cleaned datasets



'''
Created on 12-Jul-2017

@author: gayathrim
'''
