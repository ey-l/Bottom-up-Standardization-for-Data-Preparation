#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# The sinking of the Titanic is one of the most infamous shipwrecks in history.
# 
# On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.
# 
# While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.
# 
# In this challenge, we ask you to build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc).
# 
# 
# # Data Description
# 
# The training set should be used to build your machine learning models. For the training set, we provide the outcome (also known as the “ground truth”) for each passenger. Your model will be based on “features” like passengers’ gender and class. You can also use feature engineering to create new features.
# 
# The test set should be used to see how well your model performs on unseen data. For the test set, we do not provide the ground truth for each passenger. It is your job to predict these outcomes. For each passenger in the test set, use the model you trained to predict whether or not they survived the sinking of the Titanic.
# 
# We also include gender_submission.csv, a set of predictions that assume all and only female passengers survive, as an example of what a submission file should look like.
# 
# 
# Variable Notes
# pclass: A proxy for socio-economic status (SES)
# 1st = Upper
# 2nd = Middle
# 3rd = Lower
# 
# age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
# 
# sibsp: The dataset defines family relations in this way...
# 
# Sibling = brother, sister, stepbrother, stepsister
# 
# Spouse = husband, wife (mistresses and fiancés were ignored)
# 
# parch: The dataset defines family relations in this way...
# 
# Parent = mother, father
# 
# Child = daughter, son, stepdaughter, stepson
# 
# Some children travelled only with a nanny, therefore parch=0 for them.
# 
# 
# Variable	Definition	Key
# survival	Survival	0 = No, 1 = Yes
# pclass	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
# sex	Sex	
# Age	Age in years	
# sibsp	# of siblings / spouses aboard the Titanic	
# parch	# of parents / children aboard the Titanic	
# ticket	Ticket number	
# fare	Passenger fare	
# cabin	Cabin number	
# embarked	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton

# In[ ]:





# # Loading Libraries

# In[ ]:


import pandas as pd #For Data Analysis
import numpy as np # For numerical Computations
import matplotlib.pyplot as plt # For Visualization
import seaborn as sns # For Visualization
import re # For Capturing words
plt.style.use('fivethirtyeight')


# # Loading Datasets

# In[ ]:


train_df = pd.read_csv('data/input/titanic/train.csv')
test_df = pd.read_csv('data/input/titanic/test.csv')


# # Data Information and data types

# In[ ]:


# Checking the Datatypes of the columns
train_df.info()


# In[ ]:


test_df.info()


# # EDA of training data

# ## 1. Renaming columns

# In[ ]:


train_df.head()


# In[ ]:


# Converting the column names to lower_case and replacing some headings
train_df.columns = [x.lower() for x in train_df.columns]
train_df.columns


# In[ ]:


# Doing the same for test_df
test_df.columns = [x.lower() for x in test_df.columns]


# In[ ]:


train_df.rename(columns={
            "passengerid":"passenger_id",
            "pclass":"passenger_class",
            "sibsp":"sibling_spouse",
            "parch":"parent_children"
        }, inplace=True)


# In[ ]:


# Doing the same for train df
test_df.rename(columns={
            "passengerid":"passenger_id",
            "pclass":"passenger_class",
            "sibsp":"sibling_spouse",
            "parch":"parent_children"
        }, inplace=True)


# In[ ]:


train_df.head()


# ## 2. Finding Missing Values

# In[ ]:


train_df.isnull().sum()


# In[ ]:


train_df.isnull().sum().plot(kind='bar')


# In[ ]:


# Pictorial
sns.heatmap(train_df.isnull(), cbar=False)


# #### Inference (finding missing values): 
# From the above plots we can see, that Columns Age, Cabin, Embarked are missing some values. Going further we can see how we can rectify them

# ## 3. Checking Each Column values and Feature Engineering

# ### 1. Passenger Id

# In[ ]:


train_df[["passenger_id"]]


# In[ ]:


plt.figure(figsize=(12,5))
g = sns.FacetGrid(train_df, col='survived',size=5)
g = g.map(sns.distplot, "passenger_id")



# #### Inference: 
# Since passenger_id column is an index column, and it has no relation with survival rate, we can ignore the passenger_id column

# ### 2. Passenger Class

# In[ ]:


train_df.passenger_class.unique()


# #### Distribution of passenger class

# In[ ]:


train_df.passenger_class.value_counts().plot(kind='pie')


# In[ ]:


train_df.passenger_class.value_counts().plot(kind='bar')


# #### Comparison of P Class with survival

# In[ ]:


plt.figure(figsize=(12,5))
sns.countplot("passenger_class", data=train_df, hue="survived", palette="hls")
plt.ylabel("Count", fontsize=18)
plt.xlabel("P Class", fontsize=18)
plt.title("P Class Distribution ", fontsize=20)


# From the above plot, we can see Passengers in Class 1 and 2 were having good survival rate than Passenger in class 3

# In[ ]:


train_df.groupby("passenger_class").survived.value_counts(normalize=True).sort_index()


# #### Inference (passenger_class):
# From the above normalized data, we can understand that people in class 1 had 63 % survival rate and class 2 is having 47 % survival rate. 

# ### 3. Name column

# In[ ]:


train_df.name.unique()


# Name column is also like Passenger Id column, its just an index for a person.
# 
# But, from this name values, we can see different salutations are given for persons based on their age/royalty/officer.
# 
# We can collect these data first (Feature Engineering), and will analyse whether its supporting survival rate

# In[ ]:


# Collecting the salutation words
train_df.name.apply(lambda x: x.split(",")[1].split(".")[0].strip())


# In[ ]:


# Assign these values to a new column
train_df["salutation"] = train_df.name.apply(lambda x: x.split(",")[1].split(".")[0].strip())

# Doing the same for Tst data
test_df["salutation"] = test_df.name.apply(lambda x: x.split(",")[1].split(".")[0].strip())


# In[ ]:


train_df.salutation.value_counts()


# In[ ]:


#plotting countplot for salutations
plt.figure(figsize=(16,5))
sns.countplot(x='salutation', data=train_df)
plt.xlabel("Salutation", fontsize=16) 
plt.ylabel("Count", fontsize=16)
plt.title("Salutation Count", fontsize=20) 
plt.xticks(rotation=45)



# From the above graph, we can see that we have more categories in salutation, we can try to reduce it by mapping
# ( Since some categories are having only a single value, eg: Lady, Sir, Col)

# In[ ]:


# Creating Categories
salutation_dict = {
"Capt": "0",
"Col": "0",
"Major": "0",
"Dr": "0",
"Rev": "0",
"Jonkheer": "1",
"Don": "1",
"Sir" :  "1",
"the Countess":"1",
"Dona": "1",
"Lady" : "1",
"Mme": "2",
"Ms": "2",
"Mrs" : "2",
"Mlle":  "3",
"Miss" : "3",
"Mr" :   "4",
"Master": "5"
}


# In[ ]:


train_df['salutation'] = train_df.salutation.map(salutation_dict)

# Doing the same for test data
test_df['salutation'] = test_df.salutation.map(salutation_dict)


# In[ ]:


#plotting countplot for salutations
plt.figure(figsize=(16,5))
sns.countplot(x='salutation', data=train_df)
plt.xlabel("Salutation", fontsize=16) 
plt.ylabel("Count", fontsize=16)
plt.title("Salutation Count", fontsize=20) 
plt.xticks(rotation=45)



# Now we have reduced the categories

# In[ ]:


train_df.salutation = train_df.salutation.astype('float64')

# Doing the same for Test
test_df.salutation = test_df.salutation.astype('float64')


# #### Distribution of Salutation

# In[ ]:


train_df.salutation.value_counts().plot(kind='pie')


# #### Comparison with survival rate

# In[ ]:


#plotting countplot for salutations
plt.figure(figsize=(16,5))
sns.countplot(x='salutation', data=train_df, hue="survived")
plt.xlabel("Salutation", fontsize=16) 
plt.ylabel("Count", fontsize=16)
plt.title("Salutation Count", fontsize=20) 
plt.xticks(rotation=45)



# From the above plot we can see that, people in category 1, 2, 3, 5 were having mpre survival rate than other classess.
# 
# People in Category 
# 1. Jonkheer, Don, Sir, Countess, Dona, Lady
# 2. Mme, Ms, Mrs
# 3. Mlle, Miss
# 5. Master
# 
# From this we can see, Ladies and Childrens are having more survival rate. 

# In[ ]:


train_df.groupby("salutation").survived.value_counts(normalize=True).sort_index()


# In[ ]:


train_df.groupby("salutation").survived.value_counts(normalize=True).sort_index().unstack()


# So we can try to create an another column "sal_sur" based on the above findings

# In[ ]:


sal_sur_index = train_df[(train_df.salutation.isin([1.0, 2.0, 3.0, 5.0]))].index

sal_sur_index_test = test_df[(test_df.salutation.isin([1.0, 2.0, 3.0, 5.0]))].index


# In[ ]:


train_df["sal_sur"] = 0
train_df.loc[sal_sur_index, "sal_sur"] = 1

# Doing the same for test data

test_df["sal_sur"] = 0
test_df.loc[sal_sur_index_test, "sal_sur"] = 1


# In[ ]:


train_df[["sal_sur", "survived"]].head()


# In[ ]:


#plotting countplot for salutations Survived
plt.figure(figsize=(16,5))
sns.countplot(x='sal_sur', data=train_df, hue="survived")
plt.xlabel("Salutation Survived", fontsize=16) 
plt.ylabel("Count", fontsize=16)
plt.title("Salutation Survived Count", fontsize=20) 
plt.xticks(rotation=45)



# #### Inference (Name):
# From the above findings, we can see "salutations" plays a good role in survival_rate

# ### 4. Sex

# In[ ]:


# Unique values of gender
train_df.sex.unique()


# In[ ]:


# Percentage of people
train_df.sex.value_counts(normalize=True)


# #### Distribution of Gender

# In[ ]:


train_df.sex.value_counts().plot(kind='pie')


# In[ ]:


train_df.sex.value_counts().plot(kind='bar')


# In[ ]:


plt.figure(figsize=(12,5))
sns.countplot("sex", data=train_df, hue="survived", palette="hls")
plt.ylabel("Count", fontsize=18)
plt.xlabel("Sex", fontsize=18)
plt.title("Sex Distribution ", fontsize=20)


# In[ ]:


train_df.groupby("sex").survived.value_counts(normalize=True).sort_index()


# In[ ]:


train_df[['sex', 'survived']].groupby(['sex'], as_index=False).mean().sort_values(by='survived', ascending=False)


# From the above findings, we can see 74% of females are having higher survival rate than males. 

# #### Inference (Sex):
# From the above we can see females are having more survival rate than men

# ### 5. Age

# As we discussed at the top Age is having some null values. So first we can concentrate on filling the missing values first. 

# In[ ]:


train_df.age.isnull().sum()


# ### Types of filling in the data:
# 
# 1. Filling the missing data with the mean or median value if it’s a numerical variable.
# 2. Filling the missing data with mode if it’s a categorical value.
# 3. Filling the numerical value with 0 or -999, or some other number that will not occur in the data. This can be done so that the machine can recognize that the data is not real or is different.
# 4. Filling the categorical value with a new type for the missing values.
# 
# ### Process for filling missing values in Age
# 1. Since its a continous values, we can use either mean or median - Here we can use <b>Median</b>
# 2. We already having a gouping in name - like Mr, Master, Don. 
# 3. So we can group the individual name category and fill the median value to the missing items
# 

# In[ ]:


# Creating a Group based on Sex, Passenger, Salutation
age_group = train_df.groupby(["sex","passenger_class","salutation"])["age"]

# Doing the same for test data
age_group_test = test_df.groupby(["sex","passenger_class","salutation"])["age"]


# In[ ]:


# Median of each grop
age_group.median()


# In[ ]:


age_group.transform('median')


# In[ ]:


# Now we can apply the missing values
train_df.loc[train_df.age.isnull(), 'age'] = age_group.transform('median')

# Doing the same for test data
test_df.loc[test_df.age.isnull(), 'age'] = age_group_test.transform('median')


# In[ ]:


# For Checking purpose
train_df.age.isnull().sum()


# Now all the missing values are been filled. 

# #### Distribution of Age

# In[ ]:


plt.figure(figsize=(12,5))
sns.histplot(x='age', data=train_df)
plt.title("Total Distribuition and density by Age")
plt.xlabel("Age")



# In[ ]:


plt.figure(figsize=(12,5))
sns.histplot(x='age', data=train_df, hue="survived")
plt.title("Distribuition and density by Age and Survival")
plt.xlabel("Age")



# In[ ]:


plt.figure(figsize=(12,5))
sns.distplot(x=train_df.age, bins=25)
plt.title("Distribuition and density by Age")
plt.xlabel("Age")



# In[ ]:


plt.figure(figsize=(12,5))
g = sns.FacetGrid(train_df, col='survived',size=5)
g = g.map(sns.distplot, "age")



# From the above we can see that, people in the range of 18 to 40 were having good survival rate. 
# 
# Now we can see, how gender is affecting this values

# #### Male Comparisons

# In[ ]:


male_df = train_df[train_df.sex=='male']
plt.figure(figsize=(12,5))
g = sns.FacetGrid(male_df, col='survived',size=5)
g = g.map(sns.distplot, "age")



# #### Female Comparisons

# In[ ]:


female_df = train_df[train_df.sex=='female']
plt.figure(figsize=(12,5))
g = sns.FacetGrid(female_df, col='survived',size=5)
g = g.map(sns.distplot, "age")



# For Males: With age range 20 to 40 is having a good survival rate. 
# 
# For Females: With age range 18 to 40 is having a goog survival rate.
# 
# Now we can try to use this feature to build a new one 

# In[ ]:


age_index = train_df[((train_df.sex=='male') & ( (train_df.age >= 20) & (train_df.age <= 40) )) |
         ((train_df.sex=='female') & ( (train_df.age >= 18) & (train_df.age <= 40) ))   
        ].index


# In[ ]:


train_df["age_sur"] = 0
train_df.loc[age_index, "age_sur"] = 1


# In[ ]:


train_df[["age_sur","survived"]]


# In[ ]:


train_df.groupby("age_sur").survived.value_counts()


# In[ ]:


train_df["age_sur"] = 0
train_df.loc[age_index, "age_sur"] = 1
plt.figure(figsize=(12,5))
sns.countplot("age_sur", data=train_df, hue="survived", palette="hls")
plt.ylabel("Count", fontsize=18)
plt.xlabel("Age Dist", fontsize=18)
plt.title("Age Dist ", fontsize=20)


# In[ ]:


plt.figure(figsize=(12,5))
g = sns.FacetGrid(train_df, col='survived',size=5)
g = g.map(sns.distplot, "age_sur")



# Our newly created features is not good

# In[ ]:


print(sorted(train_df.age.unique()))


# In[ ]:


# We can try to create categories


# In[ ]:


interval = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 150) 
cats = list(range(len(interval)-1))

# Applying the pd.cut and using the parameters that we created 
train_df["age_category"] = pd.cut(train_df.age, interval, labels=cats)

# Printing the new Category
train_df["age_category"].head()


# Doing the same for Test Data

# Applying the pd.cut and using the parameters that we created 
test_df["age_category"] = pd.cut(test_df.age, interval, labels=cats)

# Printing the new Category
test_df["age_category"].head()


# In[ ]:


train_df.age_category.unique()


# In[ ]:


plt.figure(figsize=(12,5))
sns.countplot("age_category", data=train_df, hue="survived", palette="hls")
plt.ylabel("Count", fontsize=18)
plt.xlabel("Age Dist", fontsize=18)
plt.title("Age Dist ", fontsize=20)


# #### Comparison with Gender

# In[ ]:


male_df = train_df[train_df.sex=='male']
plt.figure(figsize=(12,5))
sns.countplot("age_category", data=male_df, hue="survived", palette="hls")
plt.ylabel("Count", fontsize=18)
plt.xlabel("Age Dist for Male", fontsize=18)
plt.title("Age Dist ", fontsize=20)


# In[ ]:


female_df = train_df[train_df.sex=='female']
plt.figure(figsize=(12,5))
sns.countplot("age_category", data=female_df, hue="survived", palette="hls")
plt.ylabel("Count", fontsize=18)
plt.xlabel("Age Dist for Female", fontsize=18)
plt.title("Age Dist ", fontsize=20)


# From the above two graphs, we can see that Males in age category 0 is having higher survival rate. 
# 
# For Female, in the range 0-6 is having higher survival rate. 
# 
# So now we can update, the new age_survival column based on our findings. 

# In[ ]:


age_index = train_df[((train_df.sex=='male') & ( train_df.age_category.isin([0]) )) |
         ((train_df.sex=='female') & ( (train_df.age_category.isin([0,1,2,3,4,5,6])) ))   
        ].index

# Doing the same for Test Data 

age_index_test = test_df[((test_df.sex=='male') & ( test_df.age_category.isin([0]) )) |
         ((test_df.sex=='female') & ( (test_df.age_category.isin([0,1,2,3,4,5,6])) ))   
        ].index


# In[ ]:


age_index


# In[ ]:


train_df["age_sur"] = 0
train_df.loc[age_index, "age_sur"] = 1

# Doing the same for Test Data 
test_df["age_sur"] = 0
test_df.loc[age_index_test, "age_sur"] = 1

plt.figure(figsize=(12,5))
sns.countplot("age_sur", data=train_df, hue="survived", palette="hls")
plt.ylabel("Count", fontsize=18)
plt.xlabel("Age Dist", fontsize=18)
plt.title("Age Dist ", fontsize=20)


# In[ ]:





# #### Inference(Age):
# From this we can know that, age_sur with category 1 is having higher survival rate

# ### 6. Sibling Spouse

# In[ ]:


train_df.sibling_spouse.unique()


# In[ ]:


train_df.groupby("sibling_spouse").survived.value_counts(normalize=True).sort_index()


# In[ ]:


plt.figure(figsize=(12,5))
sns.countplot("sibling_spouse", data=train_df, hue="survived")
plt.ylabel("Count", fontsize=18)
plt.xlabel("Sibling Dist", fontsize=18)
plt.title("Sibling Dist ", fontsize=20)


# #### Comparison with Male

# In[ ]:



male_df = train_df[train_df.sex=='male']
plt.figure(figsize=(12,5))
sns.countplot("sibling_spouse", data=male_df, hue="survived")
plt.ylabel("Count", fontsize=18)
plt.xlabel("Male Sibling Dist", fontsize=18)
plt.title("Male Sibling Dist ", fontsize=20)


# #### Comparison with Female

# In[ ]:



female_df = train_df[train_df.sex=='female']
plt.figure(figsize=(12,5))
sns.countplot("sibling_spouse", data=female_df, hue="survived")
plt.ylabel("Count", fontsize=18)
plt.xlabel("Female Sibling Dist", fontsize=18)
plt.title("Female Sibling Dist ", fontsize=20)


# #### Inference: 
# On Whole : From the above plot, we can see people with 1, 2 siblings have higher survival rate
# 
# With Male: As usual the survival rate is low for all categories
# 
# With Female : As usual the survival rate is high for all categories.

# ### 7. Parent Children

# In[ ]:


train_df.parent_children.unique()


# In[ ]:


train_df.groupby("parent_children").survived.value_counts(normalize=True).sort_index()


# In[ ]:


plt.figure(figsize=(12,5))
sns.countplot("parent_children", data=train_df, hue="survived")
plt.ylabel("Count", fontsize=18)
plt.xlabel("parent_children Dist", fontsize=18)
plt.title("parent_children Dist ", fontsize=20)


# #### Comparison with Male

# In[ ]:



male_df = train_df[train_df.sex=='male']
plt.figure(figsize=(12,5))
sns.countplot("parent_children", data=male_df, hue="survived")
plt.ylabel("Count", fontsize=18)
plt.xlabel("Male parent_children Dist", fontsize=18)
plt.title("Male parent_children Dist ", fontsize=20)


# In[ ]:


train_df[train_df.sex=='male'].groupby("parent_children").survived.value_counts(normalize=True).sort_index()


# #### Comparison with Female

# In[ ]:



female_df = train_df[train_df.sex=='female']
plt.figure(figsize=(12,5))
sns.countplot("parent_children", data=female_df, hue="survived")
plt.ylabel("Count", fontsize=18)
plt.xlabel("Female parent_children Dist", fontsize=18)
plt.title("Female parent_children Dist ", fontsize=20)


# #### Inference: 
# 
# With Male: As usual the survival rate is low for all categories
# 
# With Female : As usual the survival rate is high for all categories.

# #### We can see something is interesting right, For both sibling_spouse and parent_children, with gender as female its showing higher survival rate (in categories of 0, 1, 2, 3). 
# 
# With this information we can create a new column, like "pc_ss_sur"

# In[ ]:


ps_ss_sur_index = train_df[
    (train_df["sex"] == 'female') &
    (
        (train_df["sibling_spouse"].isin([0, 1, 2, 3])) | 
        (train_df["parent_children"].isin([0, 1, 2, 3]))
    )
].index


# Doing the same for Test Data

ps_ss_sur_index_test = test_df[
    (test_df["sex"] == 'female') &
    (
        (test_df["sibling_spouse"].isin([0, 1, 2, 3])) | 
        (test_df["parent_children"].isin([0, 1, 2, 3]))
    )
].index


# In[ ]:


train_df["ps_ss_sur"] = 0
train_df.loc[ps_ss_sur_index, "ps_ss_sur"] = 1


# In[ ]:


# Doing the same for test data

test_df["ps_ss_sur"] = 0
test_df.loc[ps_ss_sur_index_test, "ps_ss_sur"] = 1


# In[ ]:


train_df.ps_ss_sur.corr(train_df.survived)


# ### 8. Fare

# In[ ]:


print(sorted(train_df.fare.unique()))


# In[ ]:


plt.figure(figsize=(12,5))
sns.set_theme(style="whitegrid")
sns.boxplot(x="survived", y="fare", data=train_df, palette="Set3")
plt.title("Survived Fare Rate")


# In[ ]:


train_df.head()


# In[ ]:


train_df.fare.fillna(train_df.fare.mean(), inplace=True)

# Doing the same for test data
test_df.fare.fillna(test_df.fare.mean(), inplace=True)


# ### 9. Cabin

# In[ ]:


train_df.cabin.isnull().sum()


# We can see that cabin is having more of null values. So instead of filling the missing values, we can create a new feature. 

# In[ ]:


cabin_null_index = train_df[train_df.cabin.isnull()].index

# Doing the same for Cabin
cabin_null_index_test = test_df[test_df.cabin.isnull()].index


# In[ ]:


train_df["is_cabin"] = 1
train_df.loc[cabin_null_index, "is_cabin"] = 0

# Doing the same for test
test_df["is_cabin"] = 1
test_df.loc[cabin_null_index_test, "is_cabin"] = 0


# In[ ]:


train_df.is_cabin.corr(train_df.survived)


# ### 10. Embarked

# As we know before, embarked is having some missing values. We can try to fix that first. 

# In[ ]:


train_df.embarked.isnull().sum()


# In[ ]:


train_df.embarked.unique()


# #### Distribution of Embarked

# In[ ]:


train_df.embarked.value_counts().plot(kind = 'pie', autopct='%1.1f%%', figsize=(8, 8)).legend()


# In[ ]:


sns.displot(x=train_df.embarked)
plt.title("Distribuition of embarked values")



# #### Since the embarked is a categorical values, we can apply mode. So here we will be filling 'S' for all nan

# In[ ]:


train_df.embarked.fillna("S", inplace=True)


# In[ ]:


# Doing the same for test data
test_df.embarked.fillna("S", inplace=True)


# #### Survival rate based on each embarkment

# In[ ]:


sns.barplot(x='embarked', y='survived', data=train_df)


# #### Inference:
# from the above graph we can know that, people who are boarded in C were survived more

# ## Feature Scaling and Feature Selection

# In[ ]:


train_df.head()


# In[ ]:


train_df.columns


# In[ ]:


train_df.sex.replace({"male":0, "female":1}, inplace=True)


# In[ ]:


# Doing the same for test data
test_df.sex.replace({"male":0, "female":1}, inplace=True)


# In[ ]:


subset = train_df[["passenger_class", "survived","sal_sur", "age_sur", "age_category", "ps_ss_sur", "is_cabin", "sex", "fare"]]
subset_test = test_df[["passenger_class", "sal_sur", "age_sur", "age_category", "ps_ss_sur", "is_cabin", "sex", "fare"]]


# In[ ]:


subset


# In[ ]:


colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(subset.corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)


# Fare, Sex, is_Cabin, Ps_ss_sur, age_sur, sal_sur were having higher correlation with survival 

# ## Training Testing Set Preparation 

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[ ]:


X = subset.drop("survived", axis=1)
Y = train_df["survived"]


# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=10)


# ## Modelling

# In[ ]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='liblinear')