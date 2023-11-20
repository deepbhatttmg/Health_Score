#!/usr/bin/env python
# coding: utf-8

# # Predict Birth Rate from Health_Score dataset based on all other related fields.

# In[3]:


import os 
import numpy as np 
import pandas as pd


# In[9]:


os.chdir('C:\\Users\\Deep Bhatt\\Downloads\\Data Science\\Data machine\\')


# # Load Data Set 

# In[11]:


df = pd.read_excel('Health_Score.xlsx')
print(df)


# # Create X

# In[12]:


x = df.iloc[:,1:].values
print(x)


# In[13]:


y = df.iloc[:,0]
print(y)


# # Number of Records

# In[14]:


print(df.describe)


# In[15]:


print(df.info)


# In[16]:


print(df.columns)


# In[18]:


print (df.shape)

print (x.shape)

print (y.shape)


# # Train Test Split

# In[20]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=1)
print(x.shape)
print(x_train.shape)
print(y_train.shape)
print(y.shape)
print(y_train.shape)
print(y_test.shape)


# # Creating Linear Regression Model

# In[21]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
print(regressor)


# # Prediction  

# In[22]:


y_pred = regressor.predict(x_test)
print(y_pred)


# In[23]:


#Print Y test

print (y_test)


# # Display with Difference  

# In[24]:


df_y_test = pd.DataFrame(y_test,columns=['Birth Rate'])
df_y_test_pred = pd.DataFrame(y_pred,columns=['Prediction'])
df_diff = df_y_test-df_y_test_pred
y_test_pred=pd.concat([df_y_test,df_y_test_pred],axis=1)
y_test_pred['Difference']=df_y_test['Birth Rate']-df_y_test_pred['Prediction']
print(y_test_pred)


# # Accuracy of the model

# In[25]:


from sklearn.metrics import r2_score
accuracy = r2_score(y_test,y_pred)
print(accuracy)


# # Prediction   - Full Data Set 

# In[26]:


y_pred = regressor.predict(x)
print(y_pred)


# In[27]:


print(y)


# # Concatenate the perdition with Actual
# 

# In[29]:


y_pred = pd.DataFrame(y_pred,columns= ['Prediction'])
final = pd.concat([df,y_pred], axis =1)
final['Difference'] =final['Birth Rate']- final['Prediction']
display(final)


# # Accuracy of the model 

# In[30]:


from sklearn.metrics import r2_score
accuracy = r2_score(y,y_pred)
print(accuracy)


# In[31]:


#Coefficient 

print (regressor.coef_)


# In[32]:


print (regressor.intercept_)


# In[33]:


import statsmodels.api as sm
reg_ols = sm.OLS (endog = y, exog = x)
reg_ols = reg_ols.fit()
print (reg_ols.summary())

