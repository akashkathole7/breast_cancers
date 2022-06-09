#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd 
import seaborn as sns 
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


dataset = load_breast_cancer()

print(dataset)
# In[7]:





# In[8]:


data = pd.DataFrame(dataset.data)


# In[10]:


data.head()


# In[13]:


data.columns = [dataset.feature_names]


# In[15]:


data.head()


# In[16]:


data['target']= dataset.target


# In[18]:


data.head()


# In[19]:


data.isnull().sum()


# In[20]:


data.describe(include = 'all').T


# In[22]:


data.info()


# In[23]:


data.shape


# In[27]:


data.head()


# In[28]:


data['target'].value_counts()


# In[29]:


X = data.drop(columns = ['target'],axis = 1)


# In[31]:


print(X)


# In[34]:


Y = data['target']


# In[35]:


print(Y)


# In[36]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.1,random_state =1)


# In[37]:


print(X.shape,X_train.shape,X_test.shape)


# In[38]:


model = SVC()


# In[39]:


model.fit(X_train,Y_train)


# In[40]:


train_predict = model.predict(X_train)


# In[41]:


test_predict = model.predict(X_test)


# In[42]:


from sklearn.metrics import accuracy_score


# In[43]:


accuracy_score(Y_train,train_predict)


# In[44]:


accuracy_score(Y_test,test_predict)


# In[51]:


input_data = [13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259]
in_array = np.array(input_data)
reshape_data = in_array.reshape(1,-1)
predicition = model.predict(reshape_data)
if predicition[0]==1:
    print("patients have a breast cancer")
else:
    print("patients have a no breast cancer")


# In[47]:





# In[48]:




