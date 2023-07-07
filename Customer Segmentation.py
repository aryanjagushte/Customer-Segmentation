#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns                #used for visualization


# In[3]:


df = pd.read_csv('Mall_Customers.csv')


# In[4]:


df


# In[5]:


df.head(10)


# In[6]:


df.shape


# In[7]:


df.info()


# In[9]:


X = df.iloc[:,[3,4]].values   #iloc retrieves the values in the column by the index number


# In[10]:


X


# Perform Elbow method to find optimal no. of clusters

# In[11]:


from sklearn.cluster import KMeans
wcss = []          #within cluster sum of square,
#wcss decrease when k values increases, it max. at k=1


# In[12]:


for i in range(1,11):   #assuming k value between 1 and 11
    kmeans = KMeans(n_clusters=i , init='k-means++', random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)


# In[13]:


plt.plot(range(1,11),wcss)        #plot for elbow method
plt.title('The elbow method')
plt.xlabel('no. of clusters')
plt.ylabel('wcss values')


# Training a model using unsupervised learning(k-means)

# In[14]:


kmeansmodel = KMeans(n_clusters = 5, init='k-means++',random_state=0)  #from graph no. of clusters is 5


# In[15]:


y_kmeans = kmeansmodel.fit_predict(X)


# In[19]:


plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=80,c='red',label='customer 1')
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=80,c='blue',label='customer 2')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=80,c='yellow',label='customer 3')
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],s=80,c='cyan',label='customer 4')
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1],s=80,c='black',label='customer 5')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],s=100,c='magenta',label='centroids')
plt.title('clusters of customers')
plt.xlabel('annual income (k$)')
plt.ylabel('spending score (1-100)')
plt.legend()
plt.show()


# In[ ]:




