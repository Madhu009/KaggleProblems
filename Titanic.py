import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.linear_model import Perceptron
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


#Reading the data
tr=pd.read_csv('C:/Users/Madhu/Desktop/Kaggle/Titanic/Data/train.csv',header=0,sep=',')
ts=pd.read_csv('C:/Users/Madhu/Desktop/Kaggle/Titanic/Data/test.csv',header=0)


#Dropping the unnecessary columns
tr=tr.drop(['PassengerId','Name','Ticket','Embarked','Cabin'],axis=1)
ts=ts.drop(['Name','Ticket','Embarked','Cabin'],axis=1)


#Fill the 2 null values in the Embarked field
#tr['Embarked']=tr['Embarked'].fillna('S')

#Fill the missing fare value in test data
ts['Fare'].fillna(ts['Fare'].median(),inplace=True)

#Combine the Sibsp and parch into family
tr['Family']=tr['SibSp']+tr['Parch']
tr=tr.drop(['SibSp','Parch'],axis=1)
ts['Family']=ts['SibSp']+ts['Parch']
ts=ts.drop(['SibSp','Parch'],axis=1)

tr['Family'].loc[tr['Family'] > 0] = 1
tr['Family'].loc[tr['Family'] == 0] = 0

ts['Family'].loc[ts['Family'] > 0] = 1
ts['Family'].loc[ts['Family'] == 0] = 0


#Fill the age null values
tr_age_mean=tr['Age'].mean()
tr_age_std=tr['Age'].std()
tr_empty_values=tr['Age'].isnull().sum()

ts_age_mean=ts['Age'].mean()
ts_age_std=ts['Age'].std()
ts_empty_values=ts['Age'].isnull().sum()

randvalues_age_tr=np.random.randint(tr_age_mean-tr_age_std,tr_age_mean+tr_age_std,tr_empty_values)
randvalues_age_ts=np.random.randint(ts_age_mean-ts_age_std,ts_age_mean+ts_age_std,ts_empty_values)

tr['Age'][np.isnan(tr['Age'])]=randvalues_age_tr
ts['Age'][np.isnan(ts['Age'])]=randvalues_age_ts

#Convert float to int values (Fare and Age)
tr['Fare']=tr['Fare'].astype(int)
ts['Fare']=ts['Fare'].astype(int)
tr['Age']=tr['Age'].astype(int)
ts['Age']=ts['Age'].astype(int)


#Sex
def getPerson(passenger) :
    age, sex = passenger
    return 'child' if age < 16 else sex
tr['Person'] = tr[['Age', 'Sex']].apply(getPerson, axis=1)
ts['Person'] = ts[['Age', 'Sex']].apply(getPerson, axis=1)
tr.drop('Sex',axis=1,inplace=True)
ts.drop('Sex',axis=1,inplace=True)


person_dummies_train  = pd.get_dummies(tr['Person'])
person_dummies_train.columns = ['Child','Female','Male']
person_dummies_train.drop(['Male'], axis=1, inplace=True)

person_dummies_test  = pd.get_dummies(ts['Person'])
person_dummies_test.columns = ['Child','Female','Male']
person_dummies_test.drop(['Male'], axis=1, inplace=True)

tr = tr.join(person_dummies_train)
ts = ts.join(person_dummies_test)

tr.drop(['Person'], axis=1, inplace=True)
ts.drop(['Person'], axis=1, inplace=True)

#Pclass
Pclass_dummies_tr=pd.get_dummies(tr['Pclass'])
Pclass_dummies_tr.columns=['Class_1','Class_2','Class_3']
Pclass_dummies_tr.drop('Class_3',axis=1,inplace=True)

Pclass_dummies_ts=pd.get_dummies(ts['Pclass'])
Pclass_dummies_ts.columns=['Class_1','Class_2','Class_3']
Pclass_dummies_ts.drop('Class_3',axis=1,inplace=True)

tr=tr.join(Pclass_dummies_tr)
ts=ts.join(Pclass_dummies_ts)

tr.drop('Pclass',axis=1,inplace=True)
ts.drop('Pclass',axis=1,inplace=True)



#Training data
X_Train=tr.drop('Survived',axis=1)
Y_Train=tr['Survived']

X_Test=ts.drop('PassengerId',axis=1).copy()
Y_test = pd.read_csv('C:/Users/Madhu/Desktop/Kaggle/Titanic/Output/gender_submission.csv')
Y_test.drop('PassengerId', axis=1, inplace=1)
print(X_Train)

def predicAndScore(clf) :
    clf.fit(X_Train, Y_Train)
    Y_pred = clf.predict(X_Test)
    score = clf.score(X_Train, Y_Train)
    #predict_score = accuracy_score(Y_test, Y_pred)
    #print(predict_score)
    return score



#Classificaton models

# Gaussian Native Bayes
gnb = GaussianNB()
score = predicAndScore(gnb)
print('GaussianNB score:', score)

# SVM
svc = SVC()
score = predicAndScore(svc)
print('Support Vector Machine score:', score)

# Random Forests
rf = RandomForestClassifier(n_estimators=100)
score = predicAndScore(rf)
print('Random Forests:', score)

# Logistic Regression
logreg = LogisticRegression()
score = predicAndScore(logreg)
print('Logistic Regression:', score)

# knn
knn = KNeighborsClassifier(n_neighbors=3)
score = predicAndScore(knn)
print('knn:', score)

# decision tree
dt = tree.DecisionTreeClassifier()
score = predicAndScore(dt)
print('decision tree:', score)
# score: 0.957351290685
