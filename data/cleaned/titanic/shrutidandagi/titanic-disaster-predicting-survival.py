#!/usr/bin/env python
# coding: utf-8

# ![image.png](attachment:image.png)

# # <font color=blue>Domain Knowledge</font>

# 
# Titanic was a British passenger liner operated by the White Star Line. Titanic was on its way from Southampton to New York City when it sank in the North Atlantic Ocean in the early morning hours of 15 April 1912 after Titanic collided with an iceberg. The ship carried 2224 people, considering passengers and crew aboard,1514 of them died.
# 
# Titanic carried 16 wooden lifeboats and four collapsibles, which could accommodate 1,178 people, only one-third of Titanic's total capacity (and 53% of real number of passengers). At the time, lifeboats were intended to ferry survivors from a sinking ship to a rescuing ship—not keep afloat the whole population or power them to shore. If the SS Californian would responded to Titanic's distress calls, the lifeboats may have been adequate to ferry the passengers to safety as planned, but it didn't happen and the only way to survive were to get on the lifeboat.
# 
# The main question of the competition is “what sorts of people were more likely to survive?”

# # <font color=blue>Problem Statement</font>

# We are given the train and test data. The training set is used to build our machine learning models. For the training set, we are provided the outcome  for each passenger. Our model will be based on “features” like passengers’ gender and class. 
# 
# The test set is used to see how well our model performs on unseen data. For the test set, we are not provided the ground truth for each passenger. It is our job to predict these outcomes. For each passenger in the test set, use the model we trained to predict whether or not they survived the sinking of the Titanic.

# In[ ]:


# Importing neccesary libraries.

import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
sns.set(style="darkgrid")


# # <font color=blue>1. Reading and Inspection</font>

# In[ ]:


#Reading train and test files from the Titanic Data Set
train_data = pd.read_csv("data/input/titanic/train.csv")
train_data.head()


# In[ ]:


test_data = pd.read_csv("data/input/titanic/test.csv")
test_data.head()


# In[ ]:


#Determining the number of rows and columns in train and test dataframe
print('Train data: ', train_data.shape)
print('Test data: ', test_data.shape)


# > ### <font color=green>The training-set has 891 passengers and 11 features + the target variable (survived). Whereas, the test-set has 418 passengers</font>

# In[ ]:


#Lets combine both the test and train data
titanic=pd.concat([train_data, test_data], sort=False).reset_index(drop=True)
titanic.shape


# ### <font color=green>There were total 1309 passengers onboard</font>

# In[ ]:


# let's look at the statistical aspects of the dataframes
train_data.describe()


# In[ ]:


test_data.describe()


# ### <font color=green>We can see that 38% out of the training-set survived the Titanic.</font>

# In[ ]:


# Let's see the type of each column
print('Train Data')
print('-'*50)
train_data.info()
print('-'*50)
print('*'*75)
print('Test Data')
print('-'*50)
test_data.info()


# ### <font color=green>Numeric variables : PassengerId,Age,Fare,SibSp,Parch</font>
# ### <font color=green>Categorical variables : Pclass,Name,Sex,Embarked</font>

# In[ ]:


#Checking missing values in train data

print('Missing values in Train Data')
print('-'*50)
print(train_data.isnull().sum())
print('-'*50)
print('*'*75)
print('Missing Percentage in Train Data ')
print('-'*50)
print(round(100*(train_data.isnull().sum()/len(train_data.index)),2))


# ### <font color=green>Age,Cabin, Embarked features has missing  in the train data</font>

# In[ ]:


#Checking missing values in test data

print('Missing values in Test Data')
print('-'*50)
print(test_data.isnull().sum())
print('-'*50)
print('*'*75)
print('Missing Percentage in Test Data ')
print('-'*50)
print(round(100*(test_data.isnull().sum()/len(test_data.index)),2))


# ### <font color=green>Age, Cabin,Fare features has missing  in test data</font>

# # <font color=blue>2. Data Visualisation</font>

# ### <font color=blue>Visualise the data on the training data</font>

# ### <font color=Darkblue>1. Survival: </font>

# In[ ]:


plt.figure(figsize=(14, 6))
plt.subplot(1,2,1)
cols = ['r','c']



train_data['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',shadow=True, colors=cols)
plt.title('Survived')
plt.subplot(1,2,2)
sns.countplot('Survived',data=train_data,palette='PuRd')
plt.title('Survived')



# * > ### <font color=green>These plots shows that not many passengers survived the accident. Out of 891 passengers in training set, only around 350 (38.4% ) people survived of the total training set. </font>

# ### <font color=Darkblue>2. Gender: </font>

# In[ ]:


plt.figure(figsize=(14, 6))
plt.subplot(1,2,1)
cols = ['yellowgreen', 'lightcoral']
train_data['Sex'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',shadow=True, colors=cols)
plt.title('Total Male/Female onboard')
plt.subplot(1,2,2)
sns.barplot(x="Sex", y="Survived", data=train_data,palette='magma')
plt.title('Sex vs Survived')
plt.ylabel("Survival Rate")



# > ### <font color=green>In the first plot we can see, The number of men onboard are more than that of women. In the second plot we can make out that the number of women survived are more</font>
# 
# * ### Gender appears to be a very good feature to use to predict survival, as shown by the large difference in propotion survived.

# ### <font color=Darkblue>2. Pclass: </font>

# In[ ]:


plt.figure(figsize=(14, 6))
plt.subplot(1,2,1)
cols = ['gold', 'lightcoral', 'lightblue']
train_data['Pclass'].value_counts().plot.pie(autopct='%1.1f%%',shadow=True, colors=cols)
plt.title('Number of Passengers by Class')
plt.subplot(1,2,2)
sns.barplot(x='Pclass', y='Survived',data=train_data, palette='cool')
plt.title('Pclass:Survived vs Dead')
plt.ylabel("Survival Rate")



# * ### <font color=green> First plot shows, there were more passengers in Pclass 3, compared to Pclass 1 and 2</font>
# * ### <font color=green>Passenegers of Pclass 1 had a very high chances to survive.</font>
# * ### <font color=green>The number of survival from pclass 3 is low compare to Pclass 1 and 2.</font>
# 

# ### <font color=Darkblue>Survival Rates Based on Gender and Class</font>

# In[ ]:


plt.figure(figsize=(14, 8))
plt.subplot(1,2,1)
sns.barplot(x="Pclass", y="Survived", hue="Sex", data=train_data)
plt.ylabel("Survival Rate")
plt.title("Survival Rates Based on Gender and Class",fontweight="bold", size=20)

plt.subplot(1,2,2)
sns.barplot(x="Sex", y="Survived", hue="Pclass", data=train_data)
plt.ylabel("Survival Rate")
plt.title("Survival Rates Based on Gender and Class",fontweight="bold", size=20)
plt.subplots_adjust(right=1.7)




# * ### <font color=green>We can see survival rates for females is more in all the three classes compared to men</font>
# * ### <font color=green>People in Pclass 1 were more likely to survive than people in the other 2 Pclasses.</font>
# 
# 

# ### <font color=Darkblue>3. Age:   </font>

# In[ ]:


plt.figure(figsize=(14, 8))

plt.subplot(1, 2, 1)
train_data[train_data['Survived']==0].Age.plot.hist(bins=20,cmap='Set3')
plt.title("Survived",fontweight="bold", size=20)
plt.ylabel("Proportion")
plt.xlabel("Age")


plt.subplot(1, 2, 2)
train_data[train_data['Survived']==1].Age.plot.hist(bins=20, cmap='Pastel1')
plt.title("Didn't Survive",fontweight="bold", size=20)
x2=list(range(0,85,5))
plt.subplots_adjust(right=1.7)
plt.ylabel("Proportion")
plt.xlabel("Age")





# In[ ]:


#lets see these two plots together
plt.figure(figsize=(12,6))
sns.distplot(train_data[train_data['Survived']==0].Age,bins=20, kde=False, color='b', label='Died')
sns.distplot(train_data[train_data['Survived']==1].Age,bins=20, kde=False, color='r',label='Survived')
plt.title(" Age vs Survived/Died",fontweight="bold", size=20)
plt.legend()



# * > ### <font color=green>Children less than 5 years old were saved in large numbers</font>
# * > ### <font color=green>More passengers died between age 15 to 30</font>
# * > ### <font color=green>The oldest survived persen was 80.</font>

# In[ ]:


plt.figure(figsize=(14, 8))

plt.subplot(1, 2, 1)
sns.violinplot("Pclass","Age", hue="Survived", data=train_data,palette='husl')
plt.title("Pclass and Age vs Survived",fontweight="bold", size=20)
plt.subplot(1, 2, 2)
sns.violinplot("Sex","Age", hue="Survived", data=train_data,palette='Set2')
plt.title('Sex and Age vs Survived',fontweight="bold", size=20)
plt.subplots_adjust(right=1.7)



# * > ### <font color=green>Females between age 20 to 50 had high chance to survive in class 1</font>
# * > ### <font color=green>Third class young children and old aged people had less probability to survive </font>
# * > ### <font color=green>Probability of Survival is almost same for male and females between age 20 to 40</font>
# * > ### <font color=green>Females had greater chances to survive</font>

# ### <font color=Darkblue>5. Parch- Parents & Children:   </font>

# In[ ]:


plt.figure(figsize=(14, 8))

plt.subplot(1, 2, 1)
sns.barplot(x='Parch',y="Survived", data=train_data,palette='husl')
plt.title("Parent/Children vs Survival",fontweight="bold", size=20)
plt.subplot(1, 2, 2)
sns.barplot(x='Parch',y="Survived",hue='Pclass' ,data=train_data,palette='Set2')
plt.title('Parent/Children & Pclass vs Survival',fontweight="bold", size=20)
plt.subplots_adjust(right=1.7)




# * > ### <font color=green>Parents  with their children together who are one to three in number onboard have greater chance of survival. It however reduces as the number goes up</font>
# * > ### <font color=green>We can see that Parents with children together who were 1- 2 in number were in class 1 and 2 and who were three in number were in class 2</font>
# * > ### <font color=green>Class 3 has low survival rates for all the family members</font>   
# 
# 

# ### <font color=Darkblue>6.SibSp : Siblings/ Spouses   </font>

# In[ ]:


plt.figure(figsize=(14, 8))

plt.subplot(1, 2, 1)
sns.barplot(x='SibSp',y="Survived", data=train_data,palette='rocket')
plt.title("Sibling/Spouse vs Survival",fontweight="bold", size=20)
plt.subplot(1, 2, 2)
sns.barplot(x='SibSp',y="Survived",hue='Pclass' ,data=train_data,palette='rocket_r')
plt.title('Sibling/Spouse & Pclass vs Survival',fontweight="bold", size=20)
plt.subplots_adjust(right=1.7)




# * > ### <font color=green>Higher chance to survive with 1 or 2 SibSp</font>
# * > ### <font color=green>SibSp with 1,2 3 in number were in class 1 and 2</font>
# * > ### <font color=green>Class 3 has low survival rate for Siblings and spouses together</font>   
# 
# 

# ### <font color=Darkblue>7. Embarked  </font>

# In[ ]:


plt.figure(figsize=(15, 20))

plt.subplot(2, 2, 1)
sns.countplot('Embarked',data=train_data, palette='husl')
plt.title('Number of Passengers Boarded',fontweight="bold", size=20)
plt.subplot(2, 2, 2)
sns.countplot(x='Embarked',hue='Sex' ,data=train_data,palette='cool')
plt.title('Embarked: Female-Male',fontweight="bold", size=20)
plt.subplots_adjust(right=1.7)
plt.subplot(2, 2, 3)
sns.countplot(x='Embarked',hue='Pclass' ,data=train_data,palette='deep')
plt.title('Embarked vs Pclass',fontweight="bold", size=20)
plt.subplot(2, 2, 4)
sns.countplot(x='Embarked',hue='Survived' ,data=train_data,palette='rocket_r')
plt.title('Embarked vs Survived', fontweight="bold", size=20)
plt.subplots_adjust(right=1.7)



# * > ### <font color=green>Most of the passengers boarded form S and most of them were male</font>
# * > ### <font color=green>Most of the Pclass 3 boarded form S, It could be reason S has lot of died person.</font>
# 
# 
# 

# ### <font color=Darkblue>8. Fare  </font>

# In[ ]:


plt.figure(figsize=(15, 4))
plt.subplot(1,3,1)

sns.distplot(train_data[train_data['Pclass']==1].Fare, kde=False)
plt.title('Fares in Class 1')
plt.subplot(1,3,2)
sns.distplot(train_data[train_data['Pclass']==2].Fare, kde=False,color='y')
plt.title('Fares in Class 2')
plt.subplot(1,3,3)
sns.distplot(train_data[train_data['Pclass']==3].Fare, kde=False, color='g')
plt.title('Fares in Class 3')



# * > ### <font color=green>There looks to be a large distribution in the fares of Passengers in Pclass1 and this distribution goes on decreasing as the standards reduces.

# * > ## <font color=blue>Heatmap </font>

# In[ ]:


plt.figure(figsize=(12,8))
sns.heatmap(train_data.corr(),annot=True,cmap='RdYlGn')


# * >### <font color=green>We can see survival is in good relation with Fare variable</font>
# * > ### <font color=green>Survival also has strong negative correlation with sex and p-class</font>
# 

# # <font color=blue>3. Exploring some Real Data</font>

# ### Lets explore some real data on the combined data "titanic'

# In[ ]:


titanic.head()


# In[ ]:


fare=titanic.sort_values(by=['Fare'],ascending=False)
fare[0:10]


# * > ### <font color=green>We can see the highest fare was 512.39. And were in ticket class 1.</font>
# * > ### <font color=green>Highest Fare was for the people who boarded from Cherbourg</font>
# * > ### <font color=green>People who gave high fare were alone and not accompanied by siblings/Spouses and were of middle age</font>
# 

# In[ ]:


age=titanic.sort_values(by=['Age'],ascending=False)
age[0:5]


# * > ### <font color=green>Oldest passenger Survived was 80 year old boarded from Southampton who was in class 1</font>
# 

# In[ ]:


sibsp=titanic.sort_values(by=['SibSp'],ascending=False)
sibsp[0:10]


# * > ### <font color=green>Highest number of Siblings/Spouses were 8 in number, boarded from Southampton</font>
# * > ### <font color=green>None of them seems to be survived as they were in class 3.As we saw in visualisation class 3 has very less survivors</font>
# * > ### <font color=green>We can also see all 8 people were from same Sage Family</font>
# 

# In[ ]:


parch=titanic.sort_values(by=['Parch'],ascending=False)
parch[0:10]


# * > ### <font color=green>High parent children count were 9 in number</font>
# * > ### <font color=green>We can also see class Passengers having Family(SibSp, Parch were in class 3</font>
# 

# # <font color=blue>2. Data Preparation</font>

# * > ## <font color=blue>Handling missing values</font>

# * <font color=red>Cabin: Cabin has a lot of missing values. We can drop it</font>

# In[ ]:


train_data.drop("Cabin", axis = 1, inplace = True)
test_data.drop("Cabin", axis = 1, inplace = True)


# *  <font color=red>Age: We can fill in the null values with the median for the most accuracy.</font>

# In[ ]:


#the median will be an acceptable value to place in the NaN cells
train_data["Age"].fillna(train_data["Age"].median(), inplace = True)
test_data["Age"].fillna(test_data["Age"].median(), inplace = True) 


# *  <font color=red>Embarked: Embarked is a categorical variable, So here we can impute the missing values with the most popular category</font>

# In[ ]:


train_data['Embarked'].value_counts(normalize=True)


# In[ ]:


Embarked_mode=train_data['Embarked'].mode()[0]
Embarked_mode


# In[ ]:


#impute with mode
train_data["Embarked"].fillna("S", inplace = True)


# *   <font color=red>Fare: Fare in test data is a numerical variable, lets impute with median</font>

# In[ ]:


test_data["Fare"].fillna(test_data["Fare"].median(), inplace = True)


# ### <font color=green>Now lets check the missing values again</font>

# In[ ]:


#Checking missing values in train data

print('Missing values in Train Data')
print('-'*50)
print(train_data.isnull().sum())
print('-'*50)
print('*'*75)
print('Missing values in Test Data ')
print('-'*50)
print(test_data.isnull().sum())


# ### <font color=green>No more missing values in our dataset</font>

# ###  <font color=blue>Lets convert our categorical data to numeric</font>

# In[ ]:


train_data['Sex'] = train_data['Sex'].map({'female': 0, 'male': 1})
test_data['Sex']= test_data['Sex'].map({'female': 0, 'male': 1})

train_data['Embarked'] = train_data['Embarked'].map({'S': 0, 'C': 1,'Q': 2})
test_data['Embarked']= test_data['Embarked'].map({'S': 0, 'C': 1,'Q': 2})


# In[ ]:


train_data.head()


# ### We can drop Name and Ticket column 

# In[ ]:


train_data.drop(["Name","Ticket"], axis = 1, inplace = True)
test_data.drop(["Name","Ticket"], axis = 1, inplace = True)


# ### We can combine SibSp and Parch into one Family, which indicates the total number of family members on board for each member.

# In[ ]:


train_data["Family"] = train_data["SibSp"] + train_data["Parch"] + 1
test_data["Family"] = test_data["SibSp"] + test_data["Parch"] + 1


# In[ ]:


train_data.head(5)


# # <font color=blue>4. Feature Scaling</font>

# In[ ]:


scaler = StandardScaler()

train_data[['Age','Fare']] = scaler.fit_transform(train_data[['Age','Fare']])
test_data[['Age','Fare']] = scaler.transform(test_data[['Age','Fare']])

train_data.head()


# In[ ]:


test_data.head()


# # <font color=blue>4. Model Building</font>

# ### <font color=blue>Logistic Regression</font>

# In[ ]:


X_train = train_data.drop(['Survived','PassengerId'], axis=1)
y_train = train_data["Survived"]
X_test  = test_data.drop("PassengerId", axis=1)
X_train.shape, y_train.shape, X_test.shape


# In[ ]:


print(X_train.columns)
print(X_test.columns)


# In[ ]:


Features=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Family']


# In[ ]:


# Logistic Regression

LR = LogisticRegression()