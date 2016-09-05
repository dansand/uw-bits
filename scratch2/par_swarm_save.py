
# coding: utf-8

# Trying to test whether swarm saving fails in parallel if there are no particles on a proc. 

# In[1]:


import underworld as uw
import glucifer
import numpy as np


# In[9]:

mesh = uw.mesh.FeMesh_Cartesian( elementRes  = (64, 64), 
                                 minCoord    = (0., 0.), 
                                 maxCoord    = (1., 1.))


# In[10]:

swarm = uw.swarm.Swarm( mesh=mesh )


# In[ ]:




# In[ ]:




# In[11]:


# initialise a swarm
swarmCustom = uw.swarm.Swarm( mesh=mesh )
swarmCoords = np.array([ [0.2,0.2], [0.4,0.4]])
# use this array to add particles to the newly created swarm
# note that the array returned from the following method specifies the 
# local identifier for the added particles.  particles which are not 
# inside the local domain (they may live on other processors), will 
# be signified with a -1
swarmCustom.add_particles_with_coordinates(swarmCoords)


# In[16]:

testVar = uw.swarm.SwarmVariable(swarmCustom, dataType='double', count=4)
testVar.data[:] = 1.33


# In[17]:

#fig1 = glucifer.Figure()
#fig1.append( glucifer.objects.Points(swarm=swarmCustom, pointSize=5, colourBar=False) )
#fig1.append( glucifer.objects.Mesh(mesh) )

#fig1.show()


# In[20]:

print ("now we try to save")


# In[18]:

swarmCustom.save('swarm_test_save')


# In[19]:

testVar.save('var_test_save')


# In[ ]:



