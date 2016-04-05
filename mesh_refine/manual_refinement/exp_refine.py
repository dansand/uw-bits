
# coding: utf-8

# In[1]:

import numpy as np
import math
import matplotlib.pylab as pyplt
#%pylab inline


# $y = \frac{ln((\alpha x + e) -1)}{ln((\alpha + e) - 1)}$
# 

# In[124]:

#This notebook implements a logarithmic mesh refinement of the unit; we use the above function as well as its inverse
alpha = 70.
#x = np.linspace(0.00,1,100)
#y2 = (math.log(alpha*x + math.e) - 1)*(1/(math.log(alpha + math.e) - 1))

#y3 = (math.e**(x*(math.log(alpha + math.e) - 1) + 1 ) - math.e)/alpha
#pyplot.plot(x, y3)
#pyplot.plot(x, y2)
#pyplot.grid()


# In[125]:

print((math.log(alpha*0. + math.e) - 1)*(1/(math.log(alpha + math.e) - 1)), (math.log(alpha*1. + math.e) - 1)*(1/(math.log(alpha + math.e) - 1)))
print((math.e**(0*(math.log(alpha + math.e) - 1) + 1 ) - math.e)/alpha, (math.e**(1.*(math.log(alpha + math.e) - 1) + 1 ) - math.e)/alpha)

#(math.e**(0.*math.log(alpha + math.e)) - math.e)/alpha


# In[126]:

# RT PIC - classic and nearest neighbour
import underworld as uw
import math
from underworld import function as fn
import glucifer.pylab as plt
import numpy as np
import os
from shapely.geometry import Polygon
from shapely.geometry import Point


# In[127]:

dim = 2

meshX = 128
meshY = 64


# In[128]:

# create mesh objects
elementMesh = uw.mesh.FeMesh_Cartesian( elementType='Q1/dQ0', 
                                         elementRes=(meshX, meshY), 
                                           minCoord=(-1.,0.), 
                                           maxCoord=(1.,1.0))
linearMesh   = elementMesh
constantMesh = elementMesh.subMesh


# In[129]:

# create fevariables
velocityField    = uw.fevariable.FeVariable( feMesh=linearMesh,   nodeDofCount=dim )
pressureField    = uw.fevariable.FeVariable( feMesh=constantMesh, nodeDofCount=1 )
temperatureField = uw.fevariable.FeVariable( feMesh=linearMesh,   nodeDofCount=1 )


# In[130]:

for index, coord in enumerate(linearMesh.data):
    if coord[1] < 0.7:
        temperatureField.data[index] = 0.5  
    else:
        temperatureField.data[index] = 1. - coord[1]
    


# In[131]:

#(math.log(0 + 0.001)/alpha + 1) + abs((math.log(0.0 + 0.001)/alpha + 1))


# In[132]:

alpha = 20.
newys = []
newxs = []
for index, coord in enumerate(linearMesh.data):
    y0 = coord[1]
    x0 = abs(coord[0])
    newy = (math.log(alpha*y0 + math.e) - 1)*(1/(math.log(alpha + math.e) - 1))
    if coord[0] > 0:
        newx = (math.e**(x0*(math.log(alpha + math.e) - 1) + 1 ) - math.e)/alpha      
    else:
        newx = -1.*(math.e**(x0*(math.log(alpha + math.e) - 1) + 1 ) - math.e)/alpha
    newys.append(newy)
    newxs.append(newx)
    #print y0,newy


# In[133]:

linearMesh.data[:,1]


# In[134]:

with linearMesh.deform_mesh():
    linearMesh.data[:,1] = newys
    linearMesh.data[:,0] = newxs


# In[135]:


    
#figtemp = plt.Figure()
#figtemp.Surface(temperatureField, elementMesh)
#figtemp.Surface(indexField, elementMesh)
#figtemp.Mesh(linearMesh, colourBar=False)
#figtemp.show()


# In[136]:

figtemp = plt.Figure()
figtemp.Surface(temperatureField, elementMesh)
#figtemp.Surface(indexField, elementMesh)
figtemp.Mesh(linearMesh, colourBar=False)
figtemp.save_database('test_mesh_refine.gldb')


# In[137]:

#indexField = uw.fevariable.FeVariable( feMesh=linearMesh, nodeDofCount=1)

#for index, coord in enumerate(linearMesh.data):
#     indexField.data[index] = ((index % (meshX+1)) % 2 ==0)
#    indexField.data[index] = ((index/(meshX+1) % (meshY+1)) % 2 ==0)


#figindex = plt.Figure()
#figindex.Surface(indexField, elementMesh)
#figindex.Mesh(linearMesh, colourBar=False)
#figindex.show()


# In[138]:

#indexField.evaluate((0.5,0.26))


# In[139]:

#linearMesh._cself.isRegular#


# In[ ]:



