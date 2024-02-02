#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[4]:


df=pd.read_csv('countries of the world.csv')


# In[5]:


df


# In[7]:


pd.read_table('countries of the world.csv',sep=',')


# In[8]:


pd.read_json('json_sample.json')


# In[13]:


df2=pd.read_excel('world_population_excel_workbook.xlsx')


# In[14]:


df2


# In[15]:


pd.set_option('display.max.rows',235)
pd.set_option('display.max.columns',40)


# In[17]:


df2.info()


# In[19]:


df2.shape


# In[21]:


df2.head(10)


# In[22]:


df2.tail()


# In[23]:


df2['Rank']


# In[37]:


df2.loc['Rank']


# In[ ]:




