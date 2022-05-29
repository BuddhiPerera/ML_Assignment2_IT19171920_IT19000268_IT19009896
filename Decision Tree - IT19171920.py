#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#Import other necessary libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


# In[3]:


#Import the dataset
data = pd.read_csv('data.csv')


# In[4]:


#Display first 5 records of the dataset
data.head(5)


# In[5]:


#No. of rows and columns
data.shape


# In[6]:


#Infromation regarding the dataset
data.info()


# In[7]:


#Check for null values
data.isnull().sum()


# In[8]:


#Drop columns with unnecessary values
data = data.dropna(axis = 1)


# In[9]:


#No. of rows and columns
data.shape


# In[10]:


#Data types
data.dtypes


# In[11]:


#Count unique values
data['diagnosis'].value_counts()


# In[12]:


#Plot bar chart 
sns.countplot(data['diagnosis'], label = 'count')


# In[13]:


#Transform categorical to numerical
labelencoder_Y = LabelEncoder()
data.iloc[:,1] = labelencoder_Y.fit_transform(data.iloc[:,1].values)


# In[14]:


#Display
data.iloc[:,1].values


# In[15]:


#Identify Realationship
sns.pairplot(data.iloc[:,1:7], hue='diagnosis')


# In[16]:


#Correlation between columns
data.iloc[:, 1:11].corr()


# In[17]:


#Plot Heat map
plt.figure(figsize = (10, 10))
sns.heatmap(data.iloc[:,1:11].corr(), cmap="viridis", annot=True, fmt=".1%")


# In[18]:


#Feature Scaling
X = data.iloc[:,2:31].values
Y = data.iloc[:,1]


# In[19]:


#Traing and Testing dataset 80:20
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0) 


# In[20]:


#Fit the dataset into standard scaler
standardScaler = StandardScaler()
X_train = standardScaler.fit_transform(X_train)
X_test = standardScaler.fit_transform(X_test)


# In[21]:


#Display X_train values
X_train


# In[22]:


#Train the datset using decision tree 
def decisionTree(X_train, Y_train):
    tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    tree.fit(X_train, Y_train)

    #Print accuracy of the model
    print('The accuracy:', tree.score(X_train, Y_train))
    
    return tree


# In[23]:


decision_tree = decisionTree(X_train, Y_train)


# In[24]:


cm = confusion_matrix(Y_test, decision_tree.predict(X_test))
outputTP = cm[0][0]
outputTN = cm[1][1]
outputFN = cm[1][0]
outputTP = cm[0][1]

print(cm)
print('Accuracy: ', (outputTP+outputTN)/(outputTP+outputTN+outputFN+outputTP))


# In[25]:


#Summary of classification report
print(classification_report(Y_test, decision_tree.predict(X_test)))
print(accuracy_score(Y_test, decision_tree.predict(X_test)))
print()


# In[26]:


#Prediction
prediction = decision_tree.predict(X_test)
print('Model Prediction of having a Breast Cancer: ')
print(prediction)
print()
print('Actual Prediction of having a Breast Cancer: ')
print(Y_test.values)


# In[ ]:





# In[ ]:




