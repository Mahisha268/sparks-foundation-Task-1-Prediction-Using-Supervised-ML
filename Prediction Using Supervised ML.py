#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
data = pd.read_csv('set1.csv')
data.head()


# In[2]:


data.describe()


# In[3]:


data.shape

plt.scatter(data['Hours'], data['Scores'])
plt.xlabel("No. of Hours")
plt.ylabel("Scored marks")
plt.title("Change in hours and scores")
plt.show()


# In[4]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data['Hours'].values.reshape(-1,1), data['Scores'], test_size = 0.2, random_state = 42)
X_train.shape, y_train.shape, X_test.shape, y_test.shape


# In[5]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
coefficient = model.coef_
c = model.intercept_

line = (data['Hours'].values * coefficient) + c


# In[6]:


plt.scatter(data.Hours, data.Scores)
plt.plot(data.Hours, line)
plt.show()


# In[10]:


pred = model.predict(X_test)
pred

pcomp = pd.DataFrame({'Actual Values': y_test, 'Predicted Values':pred})
pcomp


# In[11]:


from sklearn import metrics
print("Mean Absolute Error: ", metrics.mean_absolute_error(y_test, pred))
print("Mean Squared Error: ", metrics.mean_squared_error(y_test, pred))
print("Root Mean Squared Error: ", metrics.mean_squared_error(y_test, pred)**0.5)
print("R2 Score: ", metrics.r2_score(y_test, pred))


# In[12]:


hours = np.asarray(9.25).reshape(-1,1)
print(f"{model.predict(hours)[0]} will be predicted score if a student study for 9.25 hrs in a day.")


# In[ ]:




