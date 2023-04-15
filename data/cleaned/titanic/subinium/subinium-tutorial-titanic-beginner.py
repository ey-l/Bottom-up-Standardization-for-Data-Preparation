#!/usr/bin/env python
# coding: utf-8

# # [수비니움의 캐글 따라하기] 타이타닉 : Beginner Ver.

# 본 커널은 다음 참고자료를 통해 재구성한 자료입니다.
# 
# - [A Journey throgh Titanic](https://www.kaggle.com/omarelgabry/a-journey-through-titanic)
# - [캐글 코리아 블로그 - 타이타닉 분석하기](https://kaggle-kr.tistory.com/17#2_6)
# 
# 저는 캐글을 시작하는 초보자이며, 초보자에게 더 적합하게 쉬운 튜토리얼을 제작하는 것을 목표로 하고 있습니다.
# 본 튜토리얼은 다음과 같은 목표 하에 제작되었습니다.
# 
# **Beginner** 
# - 데이터에 대한 정보를 최소한으로 살피며 분석을 진행합니다.
# - 어려운 메소드나 복잡한 함수의 사용을 최소화합니다.
# - 본 문제를 해결하기 위해 필수적인 요소와 순서를 서술하는 단계입니다.
# - 초심자가 접근하기에 거부감이 적어야합니다.
# - 다음 단계로 갈수록 내용이 심화됩니다.
# 
# 
# 더 많은 정보는 다음을 참고해주세요.
# 
# - **블로그** : [안수빈의 블로그](https://subinium.github.io)
# - **페이스북** : [어썸너드 수비니움](https://www.facebook.com/ANsubinium)
# - **유튜브** : [수비니움의 코딩일지](https://www.youtube.com/channel/UC8cvg1_oB-IDtWT2bfBC2OQ)

# ## 1. 라이브러리 불러오기
# 
# 우선 코드를 작성하기에 앞서 기초적으로 필요한 라이브러리를 불러옵니다.

# In[ ]:


# 필요한 라이브러리를 우선 불러옵니다.

## 데이터 분석 관련
import pandas as pd
from pandas import Series, DataFrame
import numpy as np

## 데이터 시각화 관련
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid') # matplotlib의 스타일에 관련한 함
## 그래프 출력에 필요한 IPython 명령어


## Scikit-Learn의 다양한 머신러닝 모듈을 불러옵니다.
## 분류 알고리즘 중에서 선형회귀, 서포트벡터머신, 랜덤포레스트, K-최근접이웃 알고리즘을 사용해보려고 합니다.
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


# ## 2. 데이터 읽기
# 
# kaggle 또는 데이터 분석에서 가장 많이 사용되는 파일 형식은 `csv` 파일입니다.
# 
# 코드로 데이터를 읽는 방법은 다양한 방법이 있지만, 그 중에서도 가장 유용한 것은 `pd.read_csv`로 읽는 방법입니다.

# In[ ]:


# 데이터를 우선 가져와야합니다.
train_df = pd.read_csv("data/input/titanic/train.csv")
test_df = pd.read_csv("data/input/titanic/test.csv")

# 데이터 미리보기
train_df.head()


# 위의 정보로 볼때 번호는 큰 의미를 가지지 않고, 이름과 티켓의 경우에는 불규칙성이 많아 처리하기 어려울 것 같습니다.
# 데이터의 정보는 `info` 메서드로 확인할 수 있습니다. 훈련 데이터와 테스트 데이터를 확인해보도록 하겠습니다.

# In[ ]:


train_df.info()
print('-'*20)
test_df.info()


# 위 결과에서 각각의 데이터 개수는 891개. 418개인 것을 확인할 수 있습니다.
# 특성은 각각 12개 11개입니다. 그 이유는 훈련 데이터는 생존 여부를 알고 있기 때문입니다.
# 
# 여기서 주의깊게 봐야할 부분은 다음과 같습니다.
# 
# - 각 데이터는 빈 부분이 있는가?
#     -  빈 부분이 있다면, drop할 것인가 아니면 default값으로 채워넣을 것인가
#     - cabin, Age, Embarked 세 항목에 주의
# - 데이터는 float64로 변환할 수 있는가
#     - 아니라면 범주형 데이터로 만들 수 있는가
#     
# 필요없는 부분이라고 생각되는 부분을 지웁니다. 여기서는 PassengerID와 이름, 티켓을 지웁니다.
# 이름과 티켓에서 가져올 수 있는 데이터는 없기 때문입니다. 하지만 이 문제에서 결과물은 `'PassengerId', 'Survived'` 요소가 필요하므로 훈련데이터에서만 삭제합니다.
# 

# In[ ]:


train_df = train_df.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
test_df = test_df.drop(['Name','Ticket'], axis=1)


# ## 3. 데이터 하나하나 처리하기
# 
# 이제 남은 데이터 종류는 다음과 같습니다.
# 
# 1. Pclass
# 2. Sex
# 3. Age
# 4. SibSp
# 5. Parch
# 6. Fare
# 7. Cabin
# 8. Embarked 
# 
# 이제 순서대로 보도록 하겠습니다.

# ### 3.1 Pclass
# 
# Pclass는 서수형 데이터입니다. 1등석, 2등석, 3등석과 같은 정보입니다.
# 처음에 확인시에 데이터가 비어있지 않은 것을 확인할 수 있었습니다.
# 
# 데이터에 대한 확인과 데이터를 변환해보도록 하겠습니다.
# 우선 각 unique한 value에 대한 카운팅은 `value_counts()` 메서드로 확인할 수 있습니다.

# In[ ]:


train_df['Pclass'].value_counts()


# 1,2,3은 정수이니 정수이니, 그냥 실수로만 바꾸면 되지않을까 생각할 수 있습니다.
# 하지만 1, 2, 3 등급은 경우에 따라 다를 수 있지만 연속적인 정보가 아니며, 각 차이 또한 균등하지 않습니다.
# 그렇기에 범주형(카테고리) 데이터로 인식하고 인코딩해야합니다. (비슷한 예시로 영화 별점 등이 있습니다.)
# 
# 이 데이터는 범주형 데이터이므로 one-hot-encoding을 `pd.get_dummies()` 메서드로 인코딩합시다.

# In[ ]:


pclass_train_dummies = pd.get_dummies(train_df['Pclass'])
pclass_test_dummies = pd.get_dummies(test_df['Pclass'])

train_df.drop(['Pclass'], axis=1, inplace=True)
test_df.drop(['Pclass'], axis=1, inplace=True)

train_df = train_df.join(pclass_train_dummies)
test_df = test_df.join(pclass_test_dummies)


# 이렇게 Pclass의 원본을 없애고, 범주형으로 개별로 데이터가 변환되었습니다.
# 
# > 여기서 살짝 실수한게  columns의 이름을 설정하고, 넣어줘야하는데 안그래서 1,2,3 이라는 컬럼으로 데이터가 들어갔습니다. 다른 데이터에는 이런 적용을 피하도록 합시다.
# 
# ### 3.2 Sex
# 
# Sex는 성별입니다. 남과 여로 나뉘므로 이 또한 one-hot-encoding을 진행해봅시다.

# In[ ]:


sex_train_dummies = pd.get_dummies(train_df['Sex'])
sex_test_dummies = pd.get_dummies(test_df['Sex'])

sex_train_dummies.columns = ['Female', 'Male']
sex_test_dummies.columns = ['Female', 'Male']

train_df.drop(['Sex'], axis=1, inplace=True)
test_df.drop(['Sex'], axis=1, inplace=True)

train_df = train_df.join(sex_train_dummies)
test_df = test_df.join(sex_test_dummies)


# ### 3.3 Age
# 
# 나이는 연속형 데이터이므로, 큰 처리가 필요없습니다. (카테고리화를 하여 일부 알고리즘에 더 유용한 결과를 만들 수 있습니다.)
# 하지만 일부 NaN 데이터가 있으니 이를 채울 수 있는 방법에 대해서 생각해봅시다.
# 
# 1. 랜덤
# 2. 평균값
# 3. 중간값
# 4. 데이터 버리기
# 
# 저는 일단은 평균값으로 채우도록 하겠습니다. 데이터의 통일성을 가지기 위해 train 데이터셋의 평균값으로 훈련, 테스트 데이터셋을 채우겠습니다.

# In[ ]:


train_df["Age"].fillna(train_df["Age"].mean() , inplace=True)
test_df["Age"].fillna(train_df["Age"].mean() , inplace=True)


# ### 3.4 SibSp & Panch
# 
# 형제 자매와 부모님은 가족으로 함께 처리할 수 있습니다. 하지만 마찬가지로 바꿀 필요는 없습니다.

# ### 3.5 Fare
# 
# Fare은 탑승료입니다. 신기하게 test 데이터셋에 1개의 데이터가 비어있습니다. 아마 디카프리오인듯 합니다. :-)
# 우선 빈 부분을 `fillna` 메서드로 채우겠습니다. 
# 
# 저는 데이터 누락이 아닌 무단 탑승이라 생각하고 0으로 입력하겠습니다.

# In[ ]:


test_df["Fare"].fillna(0, inplace=True)


# ### 3.6 Cabin
# 
# Cabin은 객실입니다. NaN이 대부분인 데이터이므로 버립시다. 이 데이터를 살리는 것은 너무 어려운 일입니다.

# In[ ]:


train_df = train_df.drop(['Cabin'], axis=1)
test_df = test_df.drop(['Cabin'], axis=1)


# ### 3.7 Embarked
# 
# Embarked는 탑승 항구를 의미합니다. 우선 데이터를 확인해보겠습니다.

# In[ ]:


train_df['Embarked'].value_counts()


# In[ ]:


test_df['Embarked'].value_counts()


# S가 대다수이고 일부 데이터가 비어있는 것을 알 수 있습니다. 빈 부분은 S로 우선 채우고 시작합시다. 

# In[ ]:


train_df["Embarked"].fillna('S', inplace=True)
test_df["Embarked"].fillna('S', inplace=True)


# In[ ]:


embarked_train_dummies = pd.get_dummies(train_df['Embarked'])
embarked_test_dummies = pd.get_dummies(test_df['Embarked'])

embarked_train_dummies.columns = ['S', 'C', 'Q']
embarked_test_dummies.columns = ['S', 'C', 'Q']

train_df.drop(['Embarked'], axis=1, inplace=True)
test_df.drop(['Embarked'], axis=1, inplace=True)

train_df = train_df.join(embarked_train_dummies)
test_df = test_df.join(embarked_test_dummies)


# ## 4. 데이터 나누기
# 
# 이제 학습용 데이터를 위해 데이터를 나누어야합니다.
# 
# `(정보, 생존 여부)`와 같은 형태를 위하여 다음과 같이 데이터를 나눕니다.

# In[ ]:


X_train = train_df.drop("Survived",axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId",axis=1).copy()


# ## 5. 머신러닝 알고리즘 적용하기
# 
# 이제 로지스틱 회귀, SVC, 랜덤 포레스트, K-최근접 이웃 알고리즘을 각각 적용해봅시다.

# In[ ]:


# Logistic Regression

logreg = LogisticRegression()