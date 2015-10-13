
# coding: utf-8

# In[1]:

# RT PIC - classic and nearest neighbour
import underworld as uw
import math
from underworld import function as fn
import glucifer.pylab as plt
import numpy as np
import os
import time
import h5py


# In[2]:

CASE = 2

outputPath = 'CrameriOutput/'
tempPath = 'temp/'
outputFile = 'results_case' + str(CASE) + '.dat'


# In[3]:

# make directories if they don't exist
if not os.path.isdir(outputPath):
    os.makedirs(outputPath)
if not os.path.isdir(tempPath):
    os.makedirs(tempPath) 


# In[4]:

dim = 2


# In[5]:

192*2


# In[6]:

elementMesh = uw.mesh.FeMesh_Cartesian( elementType=("Q1/dQ0"),
                                         elementRes=(384,192), 
                                           minCoord=(0.,0.), 
                                           maxCoord=(28e5,9e5)  )
linearMesh   = elementMesh
constantMesh = elementMesh.subMesh


# In[7]:

# create fevariables
velocityField    = uw.fevariable.FeVariable( feMesh=linearMesh,   nodeDofCount=dim )
pressureField    = uw.fevariable.FeVariable( feMesh=constantMesh, nodeDofCount=1 )

velocityField.data[:] = [0.,0.]
pressureField.data[:] = 0.


# In[8]:

#Boundary conditions

# Note that we use operator overloading to combine sets
IWalls = linearMesh.specialSets["MinI_VertexSet"] + linearMesh.specialSets["MaxI_VertexSet"]
JWalls = linearMesh.specialSets["MinJ_VertexSet"] + linearMesh.specialSets["MaxJ_VertexSet"]
BWalls = linearMesh.specialSets["MinJ_VertexSet"]



#free sides, no slip top
mixedslipBC = uw.conditions.DirichletCondition(     variable=velocityField, 
                                                                  nodeIndexSets=(IWalls+BWalls, JWalls)  )


# In[ ]:




# In[ ]:




# ##Geometry

# In[9]:

#sphereShape = uw.shapes.Sphere(0.5e5, 2, centre=(14e5,3e5))

from shapely.geometry import Point

#Shapely stuff

sphereshape = Point((14e5,3e5)).buffer(5e4)


# ##Particles

# In[10]:

# We create swarms of particles which can advect, and which may determine 'materials'
gSwarm = uw.swarm.Swarm( feMesh=elementMesh )

# Now we add a data variable which will store an index to determine material
materialVariable = gSwarm.add_variable( dataType="char", count=1 )

# Layouts are used to populate the swarm across the whole domain
# Create the layout object
layout = uw.swarm.layouts.GlobalSpaceFillerLayout( swarm=gSwarm, particlesPerCell=36 )
# Now use it to populate.
gSwarm.populate_using_layout(layout=layout )

# Lets initialise the 'materialVariable' data to represent two different materials. 

mantleIndex = 1
lithosphereIndex = 2
airIndex = 3
sphereIndex = 4


# Set the material to heavy everywhere via the numpy array
materialVariable.data[:] = mantleIndex


# In[11]:

for particleID in range(gSwarm.particleCoordinates.data.shape[0]):
    x = gSwarm.particleCoordinates.data[particleID][0]
    y = gSwarm.particleCoordinates.data[particleID][1]  
    if gSwarm.particleCoordinates.data[particleID][1] > 6e5 and gSwarm.particleCoordinates.data[particleID][1] < 7e5:
        materialVariable.data[particleID] =  lithosphereIndex
    elif gSwarm.particleCoordinates.data[particleID][1] > 7e5:
        materialVariable.data[particleID] =  airIndex
    elif Point(x,y).within(sphereshape):
        materialVariable.data[particleID] =  sphereIndex
    else:
        materialVariable.data[particleID] =  mantleIndex



# In[12]:

#fig1 = plt.Figure()
#fig1.Points( swarm=gSwarm, colourVariable=materialVariable )
#fig1.save_database('test_pol.gldb')
#fig1.show()


# In[13]:

incr = 5000.
xps = np.linspace(0 + 1000.,28e5 - 1000., 10000)
#yps = [7e5 + 7e3*np.cos(2*np.pi*(i/28e5)) for i in xps]
yps = [7e5 for i in xps]

surfswarm = uw.swarm.Swarm( feMesh=elementMesh )
surfswarm.add_particles_with_coordinates(np.array((xps,yps)).T)


# In[14]:

#fig2 = plt.Figure()
#fig2.Points( swarm=surfswarm, pointSize=1.0)
#fig2.Points( swarm=gSwarm, colourVariable=materialVariable )
#fig2.save_database('test_pol.gldb')
#fig2.show()


# ##Material properties

# In[15]:

print(1e23, 10.**23)


# In[16]:

#
viscosityMapFn  = fn.branching.map( keyFunc = materialVariable, 
                         mappingDict = {mantleIndex:1e21,airIndex:1e18,lithosphereIndex:1e23, sphereIndex:1e20} )


densityMapFn = fn.branching.map( keyFunc = materialVariable,
                         mappingDict = {mantleIndex:3300.,airIndex:0., lithosphereIndex:3300., sphereIndex:3200.} )

# Define our gravity using a python tuple (this will be automatically converted to a function)
if dim ==2:
    gravity = ( 0.0, -10.0 )
else:
    gravity = ( 0.0, -10.0, 0.0)
    

# now create a buoyancy force vector.. the gravity tuple is converted to a function 
# here via operator overloading

buoyancyFn = gravity*densityMapFn


# In[17]:

# Setup the Stokes system again, now with full viscosity
# For PIC style integration, we include a swarm for the a PIC integration swarm is generated within.
# For gauss integration, simple do not include the swarm. Nearest neighbour is used where required.
stokesPIC = uw.systems.Stokes(velocityField=velocityField, 
                              pressureField=pressureField,
                              conditions=[mixedslipBC,],
                              viscosityFn=fn.exception.SafeMaths(viscosityMapFn), 
                              bodyForceFn=buoyancyFn)


# In[18]:

solver = uw.systems.Solver(stokesPIC)


# In[19]:

solver.solve()


# In[20]:

# Create advector objects to advect the swarms. We specify second order integration.
advector1 = uw.systems.SwarmAdvector( swarm=gSwarm, velocityField=velocityField, order=2)
advector2 = uw.systems.SwarmAdvector( swarm=surfswarm, velocityField=velocityField, order=2)


# In[21]:

# Stepping. Initialise time and timestep.
realtime = 0.
step = 0

timevals = []
vrmsvals = []


# In[ ]:




# In[1]:

sectoka = (3600*24*365*1000.)
sectoma = (3600*24*365*1e6)


# In[23]:

# create integral to get diff 
f_o = open(outputPath+outputFile, 'w')
fname = "topo.hdf5"
fullpath = os.path.join( tempPath+ fname)
start = time.clock()
while realtime /sectoma < 22.:
    #stokesPIC2.solve(nonLinearIterate=True)
    solver.solve()
    dt1 = advector1.get_max_dt()
    dt = min((2.*sectoka),dt1)
    if step == 0:
        dt = 0.
    # Advect swarm using this timestep size
    advector1.integrate(dt)
    advector2.integrate(dt)
    # Increment
    realtime += dt
    step += 1
    timevals.append(realtime)
    #Save the suface swarm temporarily
    if uw.rank() == 0:
        surfswarm.save(fullpath)
        tempfile = h5py.File(fullpath, libver='latest')
        print tempfile.keys()
        maxt = tempfile["Position"][:][:,1].max()
        f_o.write((2*'%-15s ' + '\n') % (realtime,maxt))
        tempfile.close()
        os.remove(fullpath)
    print 'step =',step


# In[ ]:



