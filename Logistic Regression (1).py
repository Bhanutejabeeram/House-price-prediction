#!/usr/bin/env python
# coding: utf-8

# In[83]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[84]:


df=pd.read_csv('Missing Data - Sheet1.csv')


# In[85]:


df.head()


# In[86]:


df.isnull()


# In[87]:


x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values


# In[88]:


sns.heatmap(df.isnull(),cbar=False,cmap='viridis')


# In[89]:


from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
imputer.fit(df.iloc[:,1:3])
df.iloc[:, 1:3]=imputer.transform((df.iloc[:, 1:3]))
imputer.fit(x[:,1:3])
x[:,1:3]=imputer.transform((x[:,1:3]))


# In[90]:


print(df)


# In[91]:


print(x)


# In[92]:


sns.heatmap(df.isnull(),cmap='viridis',cbar=False)


# In[93]:


df.head()


# In[111]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['Country']=le.fit_transform(df['Country'])
df['Purchased']=le.fit_transform(df['Purchased'])


# In[112]:


print(df)


# In[113]:


df.columns


# In[114]:


# from sklearn.preprocessing import StandardScaler
# scaler=StandardScaler()
# scaler.fit(df.drop('Purchased',axis=1))
# scaled_features=scaler.transform(df.drop('Purchased',axis=1))
# df_scal=pd.DataFrame(scaled_features,columns=df.columns[:-1])


# In[127]:


X=df.drop('Purchased',axis=1)
y=df['Purchased']


# In[128]:


from sklearn.model_selection import train_test_split


# In[129]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)


# In[130]:


from sklearn.linear_model import LogisticRegression


# In[134]:


lr=LogisticRegression()


# In[135]:


lr.fit(X_train,y_train)


# In[136]:


predictions=lr.predict(X_test)


# In[141]:


from sklearn.metrics import confusion_matrix,accuracy_score
confusion_matrix(y_test,predictions)
accuracy=accuracy_score(y_test,predictions)
print(accuracy*100)


# In[ ]:




