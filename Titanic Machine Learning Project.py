#!/usr/bin/env python
# coding: utf-8

# In[191]:


import numpy as np 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

titanic_data = pd.read_csv('/Users/name hidden/Desktop/titanic/train.csv')
titanic_test = pd.read_csv('/Users/name hidden/Desktop/titanic/test copy 2.csv')
combine = [titanic_data, titanic_test]

print(titanic_data)
print(titanic_test)


# In[192]:


print(titanic_data.columns)


# In[193]:


print(titanic_test.columns)

#titanic_data.dropna(subset = ["Age"], inplace=True)

titanic_test.fillna(X_test.groupby(['Sex', 'Pclass']).transform('mean'))
titanic_test['Age'].fillna(value=X_test['Age'].mean(), inplace=True)#Replace Nan Values with row means
titanic_test['Fare'].fillna(value=X_test['Fare'].mean(), inplace=True)#Replace Nan Values with row means

titanic_data.fillna(X_test.groupby(['Sex', 'Pclass']).transform('mean'))
titanic_data['Age'].fillna(value=X_test['Age'].mean(), inplace=True)#Replace Nan Values with row means
titanic_data['Fare'].fillna(value=X_test['Fare'].mean(), inplace=True)#Replace Nan Values with row means



print(titanic_data)



#print(titanic_test)
#print(titanic_data)


# In[194]:


titanic_data['Age'].isnull().values.any()
titanic_test['Age'].isnull().values.any()


# In[195]:


print(titanic_data.dtypes)


# In[196]:


titanic_data['Sex'] = titanic_data['Sex'].replace(['female', 'male'], [0, 1]) 
#changing Male and Female to Binary Values
titanic_test['Sex'] = titanic_data['Sex'].replace(['female', 'male'], [0, 1])


# In[197]:


print(titanic_data.dtypes)


# In[198]:


titanic_data[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[199]:


#Women have a high survival rate of approx 74%, Sex seems to indicate survival rate and so will be included in our model


# In[200]:


titanic_data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[201]:


#Class also seems to indicate survial rate with a postiive correlation between class and survival rate


# In[202]:


var = ['Sex','Pclass','Age','Fare']
param = titanic_data[var]




X_train = param
Y_train = titanic_data["Survived"]
X_test  = titanic_test.drop("PassengerId", axis=1).copy()
X_test.fillna(X_test.groupby(['Sex', 'Pclass']).transform('mean'))
X_test['Age'].fillna(value=X_test['Age'].mean(), inplace=True)#Replace Nan Values with row means
X_test['Fare'].fillna(value=X_test['Fare'].mean(), inplace=True)#Replace Nan Values with row means

X_train.shape, Y_train.shape, X_test.shape


# In[203]:


print(param.dtypes)

X_test['Fare'].isnull().values.any()


# In[204]:


#print(X_test)

X_test.dropna(subset = ["Sex"], inplace=True)
print(X_test)

titanic_test.fillna(titanic_test.groupby(['Sex', 'Pclass']).transform('mean'))
titanic_test['Age'].fillna(value=X_test['Age'].mean(), inplace=True)#Replace Nan Values with row means
titanic_test['Fare'].fillna(value=X_test['Fare'].mean(), inplace=True)#Replace Nan Values with row means
print(titanic_test)


# In[205]:



random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2) #Score of accuracy out of 100 to 2dp
acc_random_forest


# In[206]:


print(len(Y_pred))
print(len(titanic_test.PassengerId))


# In[207]:


ml_submission = pd.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": Y_pred
    })


print(ml_submission)


# In[208]:


ml_submission.to_csv('/Users/jamallawal/Desktop/titanic/ml_submission', index=False)

