#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt 
import seaborn as sns


# In[2]:


data=pd.read_csv('Data.csv')


# In[3]:


data.head()


# In[4]:


data.describe()


# In[5]:


data.columns


# In[6]:


data.info()


# In[7]:


data.hist(figsize=(20,20))


# In[8]:


corr_matrix=data.corr()
corr_matrix['Item_Outlet_Sales']


# In[9]:


data=data.drop(['Item_Weight'],axis=1)


# In[10]:


data.Item_Identifier.value_counts()


# In[12]:


data.Item_Fat_Content.unique()


# In[13]:


data.Item_Fat_Content=data.Item_Fat_Content.replace('LF','Low Fat')
data.Item_Fat_Content=data.Item_Fat_Content.replace('low fat','Low Fat')
data.Item_Fat_Content=data.Item_Fat_Content.replace('reg','Regular')


# In[14]:


data.Item_Fat_Content.value_counts()


# In[15]:


for x in data.columns:
    if data[x].dtype=='object':
        data[x]=data[x].astype('category')


# In[16]:


data.info()


# In[26]:


fig,axes=plt.subplots(1,1,figsize=(15,10))
sns.scatterplot(x='Item_MRP',y='Item_Outlet_Sales',hue='Item_Fat_Content',data=data)


# In[27]:


fig,axes=plt.subplots(3,1,figsize=(15,15))
sns.scatterplot(x='Item_Visibility',y='Item_Outlet_Sales',hue='Item_MRP',ax=axes[0],data=data)
sns.boxplot(x='Item_Type',y='Item_Outlet_Sales',ax=axes[1],data=data)
sns.boxplot(x='Outlet_Identifier',y='Item_Outlet_Sales',ax=axes[2],data=data)


# In[28]:


fig,axes=plt.subplots(2,2,figsize=(15,12))
sns.boxplot(x='Outlet_Establishment_Year',y='Item_Outlet_Sales',ax=axes[0,0],data=data)
sns.boxplot(x='Outlet_Size',y='Item_Outlet_Sales',ax=axes[0,1],data=data)
sns.boxplot(x='Outlet_Location_Type',y='Item_Outlet_Sales',ax=axes[1,0],data=data)
sns.boxplot(x='Outlet_Type',y='Item_Outlet_Sales',ax=axes[1,1],data=data)


# In[29]:


attributes=['Item_MRP','Outlet_Type','Outlet_Location_Type','Outlet_Size','Outlet_Establishment_Year','Outlet_Identifier','Item_Type','Item_Outlet_Sales']
data=data[attributes]


# In[30]:


fig,axes=plt.subplots(2,2,figsize=(15,12))
sns.boxplot(x='Outlet_Establishment_Year',y='Item_Outlet_Sales',hue='Outlet_Size',ax=axes[0,0],data=data)
sns.boxplot(x='Outlet_Size',y='Item_Outlet_Sales',hue='Outlet_Size',ax=axes[0,1],data=data)
sns.boxplot(x='Outlet_Location_Type',y='Item_Outlet_Sales',hue='Outlet_Size',ax=axes[1,0],data=data)
sns.boxplot(x='Outlet_Type',y='Item_Outlet_Sales',hue='Outlet_Size',ax=axes[1,1],data=data)


# In[31]:


data.head()


# In[32]:


data.isnull().sum()


# In[33]:


data[data.Outlet_Size.isnull()]


# In[34]:


data.groupby(['Outlet_Location_Type','Outlet_Size'])['Outlet_Identifier'].value_counts()


# In[35]:


data.groupby('Outlet_Location_Type').Outlet_Identifier.value_counts()


# In[36]:


(data.Outlet_Identifier=='OUT010').value_counts()


# In[38]:


def FillValues(dataset):
    if dataset.Outlet_Identifier == 'OUT017' :
        dataset.Outlet_Size = 'Medium'
    elif dataset.Outlet_Identifier == 'OUT035' :
        dataset.Outlet_Size = 'Small'
    elif dataset.Outlet_Identifier == 'OUT019' :
        dataset.Outlet_Size = 'Small'
    elif dataset.Outlet_Identifier == 'OUT027' :
        dataset.Outlet_Size = 'Medium'
    elif dataset.Outlet_Identifier == 'OUT013' :
        dataset.Outlet_Size = 'High'
    elif dataset.Outlet_Identifier == 'OUT046' :
        dataset.Outlet_Size = 'Small'
    elif dataset.Outlet_Identifier == 'OUT049' :
        dataset.Outlet_Size = 'Medium'
    elif dataset.Outlet_Identifier == 'OUT018' :
        dataset.Outlet_Size = 'Medium'
    elif dataset.Outlet_Identifier == 'OUT010' :
        dataset.Outlet_Size = 'High'
    elif dataset.Outlet_Identifier == 'OUT045' :
        dataset.Outlet_Size = 'Medium'
    return(dataset)

data=data.apply(FillValues,axis=1)


# In[39]:


data.head()


# In[40]:


sns.boxplot(x='Item_Outlet_Sales',y='Outlet_Size',data=data)


# In[41]:


data_label=data.Item_Outlet_Sales
data_dum=pd.get_dummies(data.iloc[:,0:6])
data_dum['Item_Outlet_Sales']=data_label
data_dum.head()


# In[42]:


train,test = train_test_split(data_dum,test_size=0.30,random_state=0)


# In[43]:


train_label=train['Item_Outlet_Sales']
test_label=test['Item_Outlet_Sales']
del train['Item_Outlet_Sales']
del test['Item_Outlet_Sales']


# In[44]:


lr=LinearRegression()
lr.fit(train,train_label)
predict_lr=lr.predict(test)
mse=mean_squared_error(test_label,predict_lr)
lr_score=np.sqrt(mse)
lr_score


# In[45]:


from sklearn.ensemble import GradientBoostingRegressor
gbr=GradientBoostingRegressor()
gbr.fit(train,train_label)
p=gbr.predict(test)
gb_score=mean_squared_error(test_label,p)
gb_score=np.sqrt(gb_score)
gb_score


# In[46]:


from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor()
rf.fit(train,train_label)
predict_r=rf.predict(test)
mse=mean_squared_error(test_label,predict_r)
rf_score=np.sqrt(mse)
rf_score


# In[ ]:




