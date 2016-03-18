
# coding: utf-8

# In[57]:

#%matplotlib inline
import matplotlib.pyplot as plt
import underworld as uw
import glucifer
import numpy as np
import slippy2 as sp


# In[58]:

mesh = uw.mesh.FeMesh_Cartesian( elementType = ("Q1/dQ0"), 
                                 elementRes  = (256, 128), 
                                 minCoord    = (-1., 0.), 
                                 maxCoord    = (1., 1.))


# In[59]:

figMesh = glucifer.Figure(antialias=1)
#figMesh.append( glucifer.objects.Mesh(mesh.subMesh, nodeNumbers=True) )
figMesh.append( glucifer.objects.Mesh(mesh) )
figMesh.show()


# ## Shiskin refinement

# In[60]:

axis = 0
#print((mesh.minCoord[axis], mesh.maxCoord[axis], mesh.elementRes[axis]))
#print(mesh.data_nodegId.shape)
xkeys, ykeys = sp.mesh_element_keys(mesh)

#print(xkeys.shape, ykeys.shape,)


# In[61]:

testx = []
for index, coord in enumerate(mesh.data):
    testx.append(index)
    
    
    
print("unique mesh index: ", np.unique(testx).shape)


# In[ ]:




# In[62]:

#X-Axis
mesh.reset()
axis = 0
origcoords = np.linspace(mesh.minCoord[axis], mesh.maxCoord[axis], mesh.elementRes[axis] + 1)
edge_rest_lengths = np.diff(origcoords)

deform_lengths = edge_rest_lengths.copy()
third = edge_rest_lengths.shape[0]/3
deform_lengths[third:2*third] *= 0.75 ##The matix can go singular when this is exactly 1.

#print(edge_rest_lengths.shape, deform_lengths.shape)

sp.deform_1d(deform_lengths, mesh,axis = 'x',norm = 'Min', constraints = "None")


# In[63]:

#Y-Axis
axis = 1
origcoords = np.linspace(mesh.minCoord[axis], mesh.maxCoord[axis], mesh.elementRes[axis] + 1)
edge_rest_lengths = np.diff(origcoords)
third = edge_rest_lengths.shape[0]/3
deform_lengths = np.copy(edge_rest_lengths)
deform_lengths

deform_lengths[2*third:] *= 0.75 ##The matix can go singular when this is exactly 1.
sp.deform_1d(deform_lengths, mesh, axis = 'y',norm = 'Min', constraints = "None")


# In[64]:

figMesh.save_database('test.gldb')
figMesh.show()


# In[65]:

print(mesh.data_nodegId.shape, mesh.data_enMap.shape, mesh.data_elgId.shape)


# In[66]:

#


# In[ ]:




# In[41]:

#for index, coord in enumerate(mesh.data):
#    if index < mesh.data_nodegId.shape[0]:
#        print(index, mesh.data_nodegId[index][0])


# In[35]:

mesh.data.shape


# In[ ]:



