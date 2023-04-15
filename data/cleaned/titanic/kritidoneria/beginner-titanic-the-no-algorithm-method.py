#!/usr/bin/env python
# coding: utf-8

# Hi ,
# I hope you're all keeping safe.
# 
# There are so any wonderful notebooks on this dataset, that I wanted to try something funny.
# Let's not build an ML model,let's use simple rules and see how far it gets us.
# If you're a beginner, this is a great place to start.
# In ML, we start with a question. 
# 
# **Today, the question is, Given a person, did they die in Titanic disaster or not?**
# Ofcourse, the argument can be who had to die is already dead, and that is frequentist statisticians.

# <h1> Import libraries 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# <h1> Reading files </h1>

# In[ ]:


input=pd.read_csv('data/input/titanic/train.csv')
test=pd.read_csv('data/input/titanic/test.csv')


# *Survival* is the tag telling us who died and who didn't. Let's get the ball rolling.

# In[ ]:


#Subsetting columns I need
cols = ['PassengerId','Sex','Age']


# <h1> Everybody dies!! </h1>
# Let's get sadistic and say the ship froze and no one made it.

# In[ ]:


submission_all_dead = test[cols]
submission_all_dead['Survived']=0
submission_all_dead.head()


# <h1>Everybody Lives</h1>
# 
# Let's get too hopeful and say everyone left in time on lifeboats, and no one died.
# So, is [this](https://www.quora.com/Did-Leonardo-DiCaprios-character-die-at-the-end-of-Titanic-How) true?

# In[ ]:


submission_all_live = test[cols]
submission_all_live['Survived'] = 1
submission_all_live.head()


# <h1> Take a Reasonable guess </h1>
# Okay, okay. Now that I've covered the extremes, Let's agree real world is not perfect. 
# 
# The actual answer is always between the edges. Since a person cant be half alive half dead, it would mean **some** died, while **some** didn't. I've watched the movie, so I'm going to let all the women and children live for now

# In[ ]:


submission_women_child = test[cols]
submission_women_child['Survived'] = 0
#all women live
submission_women_child[submission_women_child['Sex']=='female']['Survived']=1
# all kids live
submission_women_child[submission_women_child['Age']<18]['Survived']=1


# I can keep making such edicated guesses until I'm tired.
# 
# Now the question that arises is, what if someone/something can do it for me, by looking at data?
# **That is exactly what machine learning does.**
# Hope you had fun reading this.

# <h1> Writing Submissions </h1>

# In[ ]:





