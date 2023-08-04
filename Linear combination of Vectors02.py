#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Create a vector
vec = np.array([[5],[4]])
print(vec)


# In[3]:


origin = np.zeros(vec.shape)

plt.figure(figsize=(6,6))
plt.quiver(*origin, *vec, color=['r'], scale=1, units='xy')

plt.grid()
plt.xlim(-10,10)
plt.ylim(-10,10)
plt.gca().set_aspect('equal')
plt.show()


# In[4]:


x_coord = vec[0]
y_coord = vec[1]

print("x:", x_coord)
print("y:", y_coord)


# In[5]:


# As shown in the theory above, if we multiply vectors i and j by scalar 
# we will get a vector. Let's see how we can reconstruct the original vector
# if we know the amount of movement over x and y axis.

i = np.array([[1],[0]])
j = np.array([[0],[1]])

vec_new = i*x_coord + j*y_coord
print(vec_new)


# In[6]:


# Linear combination of vectors

v = np.array([[3],[1]])
w = np.array([[-2],[-4]])


# In[7]:


a = 1.5
b = 1


# In[8]:


vec_new = a*v + b*w

print(vec_new)


# In[9]:


origin = np.zeros(vec.shape)

plt.figure(figsize=(6,6))
plt.quiver(*origin, *vec_new, color=['r'], scale=1, units='xy')
plt.quiver(*origin, *v, color=['g'], scale=1, units='xy')
plt.quiver(*origin, *w, color=['b'], scale=1, units='xy')

plt.grid()
plt.xlim(-5,5)
plt.ylim(-5,5)
plt.text(v[0], v[1], 'v')
plt.text(w[0], w[1], 'w')
plt.text(vec_new[0], vec_new[1], 'Linear combination of a*v and b*w')
plt.gca().set_aspect('equal')
plt.show()


# In[ ]:




