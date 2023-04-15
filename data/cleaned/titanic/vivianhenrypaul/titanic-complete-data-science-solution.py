#!/usr/bin/env python
# coding: utf-8

# <h1>Titanic - Complete Data Science Solution</h1>

# ![](https://cdn.britannica.com/68/185468-050-267B9304/Titanic-iceberg-British-15-1912.jpg)
# 
# Reference: [Encyclopedia Britannica](https://www.britannica.com/topic/Titanic)

# <div class = "alert alert-success">
#   <h3 style = "color:black;">Problem Statement</h3>
# </div>
# On April 15, 1912, during her maiden voyage, the "unsinkable" Titanic sank after colliding with an iceberg. This unfortunate incident resulted in the demise of 1502 out of 2224 passengers and crew.
# 
# Create a model that can determine, given a labelled training set of samples listing passengers who either did or did not survive the disaster, which of the passengers on the test dataset would have survived.
# 
# *Dataset description*
# 
# survival: Survival (0 = No, 1 = Yes)
# 
# pclass: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
# 
# sex: Sex
# 
# Age: Age in years
# 
# sibsp: # of siblings / spouses aboard the Titanic
# 
# parch: # of parents / children aboard the Titanic
# 
# ticket: Ticket number
# 
# fare: Passenger fare
# 
# cabin: Cabin number
# 
# embarked: Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

# <div class = "alert alert-success">
#   <h3 style = "color:black;">Importing relevant components</h3>
# </div>

# <h3>Libraries</h3>

# In[ ]:


# For data manipulation and visualization
import numpy as np
import pandas as pd
# import pandas_profiling
import matplotlib.pyplot as plt
import seaborn as sns

# For predictive data analysis
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_curve 


# **Note:** A common Pandas function is describe() which provides a descriptive statistical summary of all the features of the dataset. pandas_profiling is an improvement on this function and offers web format report generation for the dataset with lots of features and customizations for the report generated. However, it is not compatible with the default Python version on Kaggle, as yet, and has been commented until the issue is solved.

# <h3>Dataset</h3>
# While importing the datasets, the "PassengerId" field is made the index column.

# In[ ]:


train = pd.read_csv("data/input/titanic/train.csv", index_col = "PassengerId")
test = pd.read_csv("data/input/titanic/test.csv", index_col = "PassengerId")


# In[ ]:


train.head()


# In[ ]:


test.head()


# <div class = "alert alert-success">
#   <h3 style = "color:black;">Data wrangling</h3>
# </div>

# Data wrangling is the process of cleaning and unifying messy datasets for easy access and analysis. It is essentially the act of transforming data from a generally raw form to a more appropriate and valuable form, therbey making it suitable for a variety of downstream purposes such as analytics.
# 
# Using Pandas, we can describe the dataset and attain an in-depth understanding of the nature of our data. Using this understanding, we may then proceed to clean the data. As the popular aphorism goes... "Garbage in, garbage out." The measure of a created model will be highly dependant on the data used to create it. Thus, data wrangling is a pivotal step in the predictive data analysis pipeline. 

# **How many rows and columns do we have?**

# In[ ]:


train.shape


# In[ ]:


test.shape


# We see that the train dataset has 891 records (rows) and 11 attributes/fields (columns). The test dataset has 418 records and 10 attributes. This is expected as the test dataset will not have the label (survived/died). 

# **What features are available in the dataset?**
# 
# Features and attributes are often used interchangeably. However, to be accurate from a terminology perspective, keep in mind that attributes refer to all available fields while features refer to those attributes used to create the model.

# In[ ]:


train.columns


# In[ ]:


test.columns


# Categorical fields: These fields have a certain fixed number of valid inputs. May be nominal, ordinal, ratio-based, or interval-based.
# * Survived
# * Pclass
# * Sex
# * Embarked
# 
# Continuous fields: These fields have any number of valid inputs within theoretical minimum and maximum values. 
# * Age
# * Fare
# * SibSp
# * Parch
# 
# Understanding the data type of the various fields aids in selecting appropriate preprocessing and visualization techniques. 

# **Which features contain empty/null values?**
# 
# Model creation and employment fail when records with empty fields are passed. Thus, it becomes imperative to identify and address such records as early as possible. 

# In[ ]:


train.isna().sum()


# In[ ]:


test.isna().sum()


# It can be seen that we indeed have a considerable number of null values. In the train dataset, the "Cabin", "Age", and "Embarked" fields have 687, 177, and 2 null values respectively. In the test dataset, the "Cabin", "Age", and "Fare" fields have 327, 86, and 1 null values respectively.

# **Number of unique values in each field**

# In[ ]:


train.nunique()


# In[ ]:


test.nunique()


# **What is the data type for each field?**

# In[ ]:


train.info()


# In[ ]:


test.info()


# Looking at the dtype row gives an overview of the data types in the dataset.
# * The train dataset has 6 floating or integer fields and 5 object (string or another) fields
# * The test dataset has 5 floating or integer fields and 5 object (string or another) fields

# **What is the statistical distribution of data in each feature?**
# 
# By default, the following function only considers numerical features. To generate descriptive statistics for categorical features as well, include the optional argument 'include = [O]' in the function call.

# In[ ]:


train.describe()


# In[ ]:


train.describe(include = ["O"])


# For numerical features: 
# * Sample size is 40% of the population (891 records out of 2,224)
# * Around 38% samples survived (mean of binary class "Survived")
# * Most passengers (> 75%) did not travel with parents or children.
# * Nearly 30% of the passengers had siblings and/or spouse aboard.
# 
# For categorical features:
# * No duplicate values in "Name" (count = unique)
# * "Sex" is a binary field with 65% being male (freq. of top value (male) is 577 out of 891)
# * Embarked takes three possible values and port S port was most frequently used

# **Modifying the Cabin coloumn**
# 
# The Cabin coloumn has an unjustifiably large number of unique values. This is based on the presumption that the number allocation in the coloumn following the cabin letter has a negligible impact on a passengers probability of survival. Thus, this coloumn will be modified so as to mitigate the numbers. For records that do not have an entry in this coloumn, "na" will be assigned. 

# In[ ]:


train["Cabin"].unique()


# In[ ]:


train["Cabin"] = train["Cabin"].apply(lambda x : x[0] if pd.notna(x) else "na")
train["Cabin"].unique()


# In[ ]:


test["Cabin"] = test["Cabin"].apply(lambda x : x[0] if pd.notna(x) else "na")
test["Cabin"].unique()


# **Handling missing values**
# 
# It is not possible to create a model using a dataset with missing values. Thus, it is important to address such records. The most common approaches to handling missing values are:
# * Dropping records/coloumns in their entirety
# * Filling in missing values using appropriate method
# 
# It can be seen above that Age, Cabin, Fare, and Embarked have missing values. However, Cabin was already addressed in the step above. Thus, only the Age, Fare, and Embarked coloumns will be filled.
# 
# Additionally, here is when unnecessary fields/attributes may be dropped. 

# In[ ]:


train.reset_index(inplace = True)
test.reset_index(inplace = True)

train.drop(["PassengerId", "Name", "Ticket"], axis = 1, inplace = True)
test_passenger_ids = test["PassengerId"]  # Saved seperately for submission file
test.drop(["PassengerId", "Name", "Ticket"], axis = 1, inplace = True)

train["Age"].fillna(train["Age"].mean(skipna = True), inplace = True)
test["Age"].fillna(test["Age"].mean(skipna = True), inplace = True)

train["Embarked"].fillna("S", inplace = True)
test["Embarked"].fillna("S", inplace = True)

test["Fare"].fillna(test["Fare"].mean(skipna = True), inplace = True)


# In[ ]:


train.isna().sum()


# In[ ]:


test.isna().sum()


# **Label encoding**
# 
# Since the system can not comprehend string data, all such entries must be converted into numbers using logical techniques.

# In[ ]:


sex = {'male': 0, 'female': 1}
train["Sex"] = [sex[i] for i in train["Sex"]] 
test["Sex"] = [sex[i] for i in test["Sex"]] 

embarked = {'S': 0, 'C': 1, 'Q':2}
train["Embarked"] = [embarked[i] for i in train["Embarked"]] 
test["Embarked"] = [embarked[i] for i in test["Embarked"]] 
# train["Embarked"] = train["Embarked"].map(embarked).astype(int)
# test["Embarked"] = test["Embarked"].map(embarked).astype(int)

cabin_plot = train[["Cabin", "Survived"]]
train["Cabin"] = LabelEncoder().fit_transform(train["Cabin"])
test["Cabin"] = LabelEncoder().fit_transform(test["Cabin"])


# <div class = "alert alert-success">
#   <h3 style = "color:black;">Exploratory data analysis (EDA)</h3>
# </div>

# A picture is worth a thousand words. We will now understand the data using visualization techniques, with the help of Seaborn. 

# <div class = "alert alert-warning">
# </div>

# **Distribution of age**

# In[ ]:


plt.figure(figsize = (20,10))
sns.histplot(x = "Age", data = train)
plt.title("Histogram (Age)")



# In[ ]:


plt.figure(figsize = (20,10))
sns.kdeplot(x = "Age", data = train, fill = True)
plt.title("KDE (Age)")



# In[ ]:


plt.figure(figsize = (20,2))
sns.boxplot(x = "Age", data = train)
plt.title("Boxplot (Age)")



# In[ ]:


plt.figure(figsize = (20,2))
sns.violinplot(x = "Age", data = train)
plt.title("Violin plot (Age)")



# <div class = "alert alert-warning">
# </div>

# **Distribution of fare**

# In[ ]:


plt.figure(figsize = (20,10))
sns.histplot(x = "Fare", data = train)
plt.title("Histogram (Fare)")



# In[ ]:


plt.figure(figsize = (20,10))
sns.kdeplot(x = "Fare", data = train, fill = True)
plt.title("KDE (Fare)")



# In[ ]:


plt.figure(figsize = (20,2))
sns.boxplot(x = "Fare", data = train)
plt.title("Boxplot (Fare)")



# In[ ]:


plt.figure(figsize = (20,2))
sns.violinplot(x = "Fare", data = train)
plt.title("Violin plot (Fare)")



# <div class = "alert alert-warning">
# </div>

# **Distribution of gender**

# In[ ]:


plt.figure(figsize = (10, 5))
sns.countplot(x = "Sex", data = train)
plt.title("Countplot (Sex)")
plt.xticks([0, 1], ["Male", "Female"])



# <div class = "alert alert-warning">
# </div>

# **Relation of various features with Survived**

# In[ ]:


plt.figure(figsize = (20, 10))
sns.countplot(x = "Pclass", hue = "Survived", data = train)
plt.title("Survived VS Pclass")



# In[ ]:


plt.figure(figsize = (10, 5))
train.loc[train["Sex"] == 0, "Survived"].value_counts().plot.pie(autopct="%1.1f%%")
plt.title("Percentage survived (Male)")



# In[ ]:


plt.figure(figsize = (10, 5))
train.loc[train["Sex"] == 1, "Survived"].value_counts().plot.pie(autopct="%1.1f%%")
plt.title("Percentage survived (Female)")



# In[ ]:


plt.figure(figsize = (20, 10))
sns.countplot(x = "SibSp", hue = "Survived", data = train)
plt.title("Survived VS SibSp")



# In[ ]:


plt.figure(figsize = (20, 10))
sns.countplot(x = "Cabin", hue = "Survived", data = cabin_plot)
plt.title("Survived VS Cabin")



# In[ ]:


plt.figure(figsize = (20, 10))
sns.kdeplot(x = "Age", hue = "Survived", data = train, shade = True)
plt.title("Survived VS Age")



# In[ ]:


plt.figure(figsize = (20, 10))
sns.kdeplot(x = "Fare", hue = "Survived", data = train, shade = True)
plt.title("Survived VS Fare")



# <div class = "alert alert-warning">
# </div>

# **Relation of various features with Age and Survived**

# In[ ]:


fig, axes = plt.subplots(3, 2, figsize = (20, 20))

sns.violinplot(x = "Pclass", y = "Age", hue = "Survived", split = True, data = train, ax = axes[0, 0])
sns.violinplot(x = "Sex", y = "Age", hue = "Survived", split = True, data = train, ax = axes[0, 1])
sns.violinplot(x = "SibSp", y = "Age", hue = "Survived", split = True, data = train, ax = axes[1, 0])
sns.violinplot(x = "Parch", y = "Age", hue = "Survived", split = True, data = train, ax = axes[1, 1])
sns.violinplot(x = "Embarked", y = "Age", hue = "Survived", split = True, data = train, ax = axes[2, 0])
sns.violinplot(x = "Cabin", y = "Age", hue = "Survived", split = True, data = train, ax = axes[2, 1])




# <div class = "alert alert-warning">
# </div>

# <div class = "alert alert-success">
#   <h3 style = "color:black;">Model creation and evaluation</h3>
# </div>

# There is a wide array of predictive modeling algorithms at our disposal. The algorithm(s) chosen is contingent on the problem itself and solution requirement. The problem posed here is a classification problem, in that the model must use the features provided to classify an unknown passenger as a survivior (1) or victim (0). Additionally, the category of machine learning seen here is supervised learning, since we have a labelled training dataset from which the model is created. Taking these points into consideration, we may decide on few algorithms to implement. The ones implemented here are:
# 
# 1. Decision tree
# 2. Random forest
# 3. Naive Bayes
# 4. K-nearest neighbours
# 5. Support vector machine

# In[ ]:


y = train["Survived"]
X = train.drop("Survived", axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# **Decision tree classifier**
# 
# It is a tree-structured classifier, where internal nodes represent the features of a dataset, branches represent the decision rules and each leaf node represents the outcome. In a Decision tree, there are two nodes, which are the Decision Node and Leaf Node. There are generally two types of decision trees. Models where the target variable can take a finite set of values are called classification trees. Trees where the target variable can take continuous values (typically real numbers) are called regression trees.
# 
# ![](https://cdn-images-1.medium.com/max/824/0*J2l5dvJ2jqRwGDfG.png)

# In[ ]:


decision_tree_model = DecisionTreeClassifier(random_state = 2)