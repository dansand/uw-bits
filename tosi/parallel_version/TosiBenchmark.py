
# coding: utf-8

# 
# Viscoplastic thermal convection in a 2-D square box
# =======
# 
# Benchmarks from Tosi et al. 2015
# --------
# 
# 

# This notebook generates models from the <a name="ref-1"/>[(Tosi et al., 2015)](#cite-tosi2015community) in Underworld2. The Underworld2 results are compared to the model run on Fenics. Input files for the Fenics models were provided by Petra Maierova.
# 
# This example uses the RT PIC solver with classic and nearest neighbour
# 
# 
# References
# ====
# 
# <a name="cite-tosi2015community"/><sup>[^](#ref-1) </sup>Tosi, Nicola and Stein, Claudia and Noack, Lena and H&uuml;ttig, Christian and Maierov&aacute;, Petra and Samuel, Henri and Davies, DR and Wilson, CR and Kramer, SC and Thieulot, Cedric and others. 2015. _A community benchmark for viscoplastic thermal convection in a 2-D square box_.
# 
# 

# Load python functions needed for underworld. Some additional python functions from os, math and numpy used later on.

# In[1]:

import underworld as uw
import math
from underworld import function as fn
import glucifer
import matplotlib.pyplot as pyplot
import time
import numpy as np
import os


# Set physical constants and parameters, including the Rayleigh number (*RA*). 

# In[2]:

#Do you want to write hdf5 files - Temp, RMS, viscosity, stress?
writeFiles = False


# In[3]:

#ETA_T = 1e5
#newvisc = math.exp(math.log(ETA_T)*0.64)
#newvisc


# In[4]:

case_dict = {}
case_dict[1] = {}
case_dict[1]['ETA_Y'] = 1.
case_dict[1]['YSTRESS'] = 1.

case_dict[2] = {}
case_dict[2]['ETA_Y'] = 1.
case_dict[2]['YSTRESS'] = 1.

case_dict[3] = {}
case_dict[3]['ETA_Y'] = 10.

case_dict[4] = {}
case_dict[4]['ETA_Y'] = 10.
case_dict[4]['YSTRESS'] = 1.

case_dict[5] = {}
case_dict[5]['ETA_Y'] = 10.
case_dict[5]['YSTRESS'] = 4.


# In[5]:

CASE = 4 # select identifier of the testing case (1-5)


# In[6]:

RA  = 1e2        # Rayleigh number
TS  = 0          # surface temperature
TB  = 1          # bottom boundary temperature (melting point)
ETA_T = 1e5
ETA_Y = case_dict[CASE]['ETA_Y']
ETA0 = 1e-3
TMAX = 3.0
IMAX = 1000
YSTRESS = case_dict[CASE]['YSTRESS']
RES = 40


# Alternatively, use a reference viscosity identical to Crameri and Tackley, i.e. normalised at 0.64.
# In this case the Rayligh number would go to:

# In[7]:

math.exp(math.log(10**5)*0.64)


# In[8]:

#RA2/RA == newvisc


# In[9]:

#RA2 = RA *math.exp(math.log(10**5)*0.64)
#print('new Rayleigh number would be: ' + str(RA2))


# In[10]:

#print(str(RA2), str(ETA0), str(YSTRESS))


# Simulation parameters. Resolution in the horizontal (*Xres*) and vertical (*Yres*) directions.

# In[11]:


dim = 2          # number of spatial dimensions


# Select which case of viscosity from Tosi et al (2015) to use. Adjust the yield stress to be =1 for cases 1-4, or between 3.0 and 5.0 (in increments of 0.1) in case 5.

# Set output file and directory for results

# In[12]:

outputPath = 'TosiOutput/'
imagePath = 'TosiOutput/images'
filePath = 'TosiOutput/files'
dbPath = 'TosiOutput/gldbs'
outputFile = 'results_case' + str(CASE) + '.dat'


# make directories if they don't exist
if not os.path.isdir(outputPath):
    os.makedirs(outputPath)
if not os.path.isdir(imagePath):
    os.makedirs(imagePath)
if not os.path.isdir(dbPath):
    os.makedirs(dbPath)
if not os.path.isdir(filePath):
    os.makedirs(filePath)


# Create mesh objects. These store the indices and spatial coordiates of the grid points on the mesh.

# In[13]:

elementMesh = uw.mesh.FeMesh_Cartesian( elementType=("Q1/dQ0"), 
                                         elementRes=(RES, RES), 
                                           minCoord=(0.,0.), 
                                           maxCoord=(1.,1.)  )
linearMesh   = elementMesh
constantMesh = elementMesh.subMesh 


# Create Finite Element (FE) variables for the velocity, pressure and temperature fields. The last two of these are scalar fields needing only one value at each mesh point, while the velocity field contains a vector of *dim* dimensions at each mesh point.

# In[14]:

velocityField    = uw.fevariable.FeVariable( feMesh=linearMesh,   nodeDofCount=dim )
pressureField    = uw.fevariable.FeVariable( feMesh=constantMesh, nodeDofCount=1 )
temperatureField = uw.fevariable.FeVariable( feMesh=linearMesh,   nodeDofCount=1 )
temperatureDotField = uw.fevariable.FeVariable( feMesh=linearMesh,      nodeDofCount=1 )


# Create some dummy fevariables for doing top and bottom boundary calculations.

# In[15]:

topField    = uw.fevariable.FeVariable( feMesh=linearMesh,   nodeDofCount=1)
bottomField    = uw.fevariable.FeVariable( feMesh=linearMesh,   nodeDofCount=1)

topField.data[:] = 0.
bottomField.data[:] = 0.

# lets ensure temp boundaries are still what we want 
# on the boundaries
for index in linearMesh.specialSets["MinJ_VertexSet"]:
    bottomField.data[index] = 1.
for index in linearMesh.specialSets["MaxJ_VertexSet"]:
    topField.data[index] = 1.


# #ICs and BCs

# In[16]:

# Initialise data.. Note that we are also setting boundary conditions here
velocityField.data[:] = [0.,0.]
pressureField.data[:] = 0.
temperatureField.data[:] = 0.
temperatureDotField.data[:] = 0.


# Setup temperature initial condition via numpy arrays
A = 0.01
#Note that width = height = 1
tempNump = temperatureField.data
for index, coord in enumerate(linearMesh.data):
    pertCoeff = (1- coord[1]) + A*math.cos( math.pi * coord[0] ) * math.sin( math.pi * coord[1] )
    tempNump[index] = pertCoeff;
    


# In[17]:

# Get the actual sets 
#
#  HJJJJJJH
#  I      I
#  I      I
#  I      I
#  HJJJJJJH
#  
#  Note that H = I & J 

# Note that we use operator overloading to combine sets
IWalls = linearMesh.specialSets["MinI_VertexSet"] + linearMesh.specialSets["MaxI_VertexSet"]
JWalls = linearMesh.specialSets["MinJ_VertexSet"] + linearMesh.specialSets["MaxJ_VertexSet"]
TWalls = linearMesh.specialSets["MaxJ_VertexSet"]
BWalls = linearMesh.specialSets["MinJ_VertexSet"]


# In[18]:

# Now setup the dirichlet boundary condition
# Note that through this object, we are flagging to the system 
# that these nodes are to be considered as boundary conditions. 
# Also note that we provide a tuple of sets.. One for the Vx, one for Vy.
freeslipBC = uw.conditions.DirichletCondition(     variable=velocityField, 
                                              nodeIndexSets=(IWalls,JWalls) )

# also set dirichlet for temp field
tempBC = uw.conditions.DirichletCondition(     variable=temperatureField, 
                                              nodeIndexSets=(JWalls,) )


# In[19]:

# Set temp boundaries 
# on the boundaries
for index in linearMesh.specialSets["MinJ_VertexSet"]:
    temperatureField.data[index] = TB
for index in linearMesh.specialSets["MaxJ_VertexSet"]:
    temperatureField.data[index] = TS


# #Material properties
# 

# In[20]:

#Make variables required for plasticity

secinvCopy = fn.tensor.second_invariant( 
                    fn.tensor.symmetric( 
                        velocityField.gradientFn ))


# In[21]:

coordinate = fn.input()


# In[22]:

#Remember to use floats everywhere when setting up functions

#Linear viscosities
#viscosityl1 = fn.math.exp((math.log(ETA_T)*-1*temperatureField) + (math.log(ETA_T)*-1*0.64))
viscosityl1 = fn.math.exp(math.log(ETA_T)*-1.*temperatureField)

viscosityl2 = fn.math.exp((math.log(ETA_T)*-1.*temperatureField) + (1.-coordinate[1])*math.log(ETA_Y))

viscosityFn1 = viscosityl1 #This one always gets passed to the first velcotity solve

#Von Mises effective viscosity
viscosityp = ETA0 + YSTRESS/(secinvCopy/math.sqrt(0.5)) #extra factor to account for underworld second invariant form


if CASE == 1:
    viscosityFn2 = viscosityFn1
elif CASE == 2:
    viscosityFn2 = 2./(1./viscosityl1 + 1./viscosityp)
elif CASE == 3:
    viscosityFn2 = viscosityl2
else:
    viscosityFn2 = 2./(1./viscosityl2 + 1./viscosityp)


# In[23]:

print(RA, ETA_T, ETA_Y, ETA0, YSTRESS)


# Set up simulation parameters and functions
# ====
# 
# Here the functions for density, viscosity etc. are set. These functions and/or values are preserved for the entire simulation time. 

# In[24]:

densityFn = RA*temperatureField

# define our vertical unit vector using a python tuple (this will be automatically converted to a function)
z_hat = ( 0.0, 1.0 )

# now create a buoyancy force vector using the density (FEvariable) and the vertical unit vector. 
# The result from which will also be a FEvariable.
buoyancyFn = z_hat * densityFn 


# Build the Stokes system, solvers, advection-diffusion
# ------
# 
# Setup linear Stokes system to get the initial velocity.

# In[25]:

#We first set up a l
stokesPIC = uw.systems.Stokes(velocityField=velocityField, 
                              pressureField=pressureField,
                              conditions=[freeslipBC,],
#                              viscosityFn=viscosityFn1, 
                              viscosityFn=fn.exception.SafeMaths(viscosityFn1), 
                              bodyForceFn=buoyancyFn)


# We do one solve with linear viscosity to get the initial strain rate invariant. This solve step also calculates a 'guess' of the the velocity field based on the linear system, which is used later in the non-linear solver.

# In[26]:

stokesPIC.solve()


# In[27]:

# Setup the Stokes system again, now with linear or nonlinear visocity viscosity.
stokesPIC2 = uw.systems.Stokes(velocityField=velocityField, 
                              pressureField=pressureField,
                              conditions=[freeslipBC,],
                              viscosityFn=fn.exception.SafeMaths(viscosityFn2), 
                              bodyForceFn=buoyancyFn )


# In[28]:

solver = uw.systems.Solver(stokesPIC2) # altered from PIC2


# Solve for initial pressure and velocity using a quick non-linear Picard iteration
# 

# In[29]:

solver.solve(nonLinearIterate=True)


# Create an advective-diffusive system
# =====
# 
# Setup the system in underworld by flagging the temperature and velocity field variables.

# In[30]:

# Create advdiff system
#advDiff = uw.systems.AdvectionDiffusion( temperatureField, velocityField, diffusivity=1., conditions=[tempBC,] )
advDiff = uw.systems.AdvectionDiffusion( temperatureField, temperatureDotField, velocityField, diffusivity=1., conditions=[tempBC,] )



# Metrics for benchmark
# =====
# 
# Define functions to be used in the time loop. For cases 1-4, participants were asked to report a number of diagnostic quantities to be measured after reaching steady state:
# 
# * Average temp... $$  \langle T \rangle  = \int^1_0 \int^1_0 T \, dxdy $$
# * Top and bottom Nusselt numbers... $$N = \int^1_0 \frac{\partial T}{\partial y} \rvert_{y=0/1} \, dx$$
# * RMS velocity over the whole domain, surface and max velocity at surface
# * max and min viscosity over the whole domain
# * average rate of work done against gravity...$$\langle W \rangle = \int^1_0 \int^1_0 T u_y \, dx dy$$
# * and the average rate of viscous dissipation...$$\langle \Phi \rangle = \int^1_0 \int^1_0 \tau_{ij} \dot \epsilon_{ij} \, dx dy$$
# 
# * In steady state, if thermal energy is accurately conserved, the difference between $\langle W \rangle$ and $\langle \Phi \rangle / Ra$ must vanish, so also reported is the percentage error: 
# 
# $$ \delta = \frac{\lvert \langle W \rangle - \frac{\langle \Phi \rangle}{Ra} \rvert}{max \left(  \langle W \rangle,  \frac{\langle \Phi \rangle}{Ra}\right)} \times 100% $$

# In[31]:

#Setup some Integrals. We want these outside the main loop...
tempint = uw.utils.Integral(temperatureField, linearMesh)
areaint = uw.utils.Integral(1.,linearMesh)

v2int = uw.utils.Integral(fn.math.dot(velocityField,velocityField), linearMesh)
topareaint = uw.utils.Integral((topField*1.),linearMesh)

dwint = uw.utils.Integral(temperatureField*velocityField[1], linearMesh)

secinv = fn.tensor.second_invariant(
                    fn.tensor.symmetric(
                        velocityField.gradientFn ))

sinner = fn.math.dot(secinv,secinv)
vdint = uw.utils.Integral((4.*viscosityFn2*sinner), linearMesh)


# In[32]:

def avg_temp():
    return tempint.evaluate()[0]/areaint.evaluate()[0]


def nusseltNumber(temperatureField, temperatureMesh, indexSet):
    tempgradField = temperatureField.gradientFn
    vertGradField = tempgradField[1]
    Nu = uw.utils.Integral(vertGradField , mesh=temperatureMesh, integrationType='Surface', surfaceIndexSet=indexSet)
    return Nu.evaluate()[0]


def rms():
    return math.sqrt(v2int.evaluate()[0])


#This one gets cleaned up when Surface integrals are available
def rmsBoundary(velocityField, velocityMesh, indexSet):
    v2fn = fn.math.dot(velocityField,velocityField)
    v2int = uw.utils.Integral(v2fn, mesh=velocityMesh, integrationType='Surface', surfaceIndexSet=indexSet)
    rmsbound = math.sqrt(v2int.evaluate()[0]) 
    return rmsbound
    
    
    fn.math.dot(velocityField,np.array([1.,0.]))
    return math.sqrt(v2int.evaluate()[0]/topareaint.evaluate()[0])

def max_vx_surf(velfield, mesh):
    vuvelxfn = fn.view.min_max(velfield[0])
    vuvelxfn.evaluate(mesh.specialSets["MaxJ_VertexSet"])
    return vuvelxfn.max_global()

def gravwork(workfn):
    return workfn.evaluate()[0]

def viscdis(vdissfn):
    return vdissfn.evaluate()[0]

def visc_extr(viscfn):
    vuviscfn = fn.view.min_max(viscfn)
    vuviscfn.evaluate(linearMesh)
    return vuviscfn.max_global(), vuviscfn.min_global()


# In[ ]:




# In[33]:

#Fields for saving data / fields

rmsField = uw.fevariable.FeVariable( feMesh=linearMesh,   nodeDofCount=1)
rmsfn = fn.math.sqrt(fn.math.dot(velocityField,velocityField))
rmsdata = rmsfn.evaluate(linearMesh)
rmsField.data[:] = rmsdata 

viscField = uw.fevariable.FeVariable( feMesh=linearMesh,   nodeDofCount=1)
viscdata = viscosityFn2.evaluate(linearMesh)
viscField.data[:] = viscdata


stressField = uw.fevariable.FeVariable( feMesh=linearMesh,   nodeDofCount=1)
srtdata = fn.tensor.second_invariant( 
                    fn.tensor.symmetric( 
                        velocityField.gradientFn ))
rostfield = srtdata.evaluate(linearMesh)
stressinv = 2*viscdata*rostfield[:]
stressField.data[:] = stressinv


# Main simulation loop
# =======
# 
# The main time stepping loop begins here. Before this the time and timestep are initialised to zero and the output statistics arrays are set up. Also the frequency of outputting basic statistics to the screen is set in steps_output.
# 

# In[34]:

realtime = 0.
step = 0
timevals = [0.]
steps_end = 5
steps_output = 100
steps_display_info = 20


# In[35]:

# initialise timer for computation
start = time.clock()
# setup summary output file (name above)
f_o = open(outputPath+outputFile, 'w')
# Perform steps
#while realtime < 0.15:
while step < steps_end:
    #Enter non-linear loop
    solver.solve(nonLinearIterate=True)
    dt = advDiff.get_max_dt()
    if step == 0:
        dt = 0.
    # Advect using this timestep size   
    advDiff.integrate(dt)
    # Increment
    realtime += dt
    step += 1
    timevals.append(realtime)
    # Calculate the Metrics, only on 1 of the processors:
    Avg_temp = avg_temp()
    Rms = rms()
    Rms_surf = rmsBoundary(velocityField,linearMesh, TWalls)
    Max_vx_surf = max_vx_surf(velocityField, linearMesh)
    Gravwork = gravwork(dwint)
    Viscdis = viscdis(vdint)
    nu0, nu1 = nusseltNumber(temperatureField, linearMesh, BWalls), nusseltNumber(temperatureField, linearMesh, TWalls)
    etamax, etamin = visc_extr(viscosityFn2)
    # output to summary text file
    if uw.rank()==0:
        f_o.write((11*'%-15s ' + '\n') % (realtime, Viscdis, nu0, nu1, Avg_temp, Rms,Rms_surf,Max_vx_surf,Gravwork, etamax, etamin))
   
    if step %  steps_display_info == 0:
        print('steps = {0:6d}; time = {1:.3e}; v_rms = {2:.3f}; Nu0 = {3:.3f}; Nu1 = {3:.3f}'
          .format(step, realtime, Rms, nu0, nu1))
    # output image to file
    if (step % steps_output == 0) & (writeFiles == True):
        ##Files to save
        #Temp
        fnametemp = "temperatureField" + "_" + str(CASE) + "_" + str(step) + ".hdf5"
        fullpath = os.path.join(outputPath + "files/" + fnametemp)
        temperatureField.save(fullpath)
        #RMS
        fnamerms = "rmsField" + "_" + str(CASE) + "_" + str(step) + ".hdf5"
        fullpath = os.path.join(outputPath + "files/" + fnamerms)
        rmsField.save(fullpath)
        #Viscosity
        fnamevisc = "viscField" + "_" + str(CASE) + "_" + str(step) + ".hdf5"
        fullpath = os.path.join(outputPath + "files/" + fnamevisc)
        viscField.save(fullpath)
        #Stress
        fnamestress = "stressField" + "_" + str(CASE) + "_" + str(step) + ".hdf5"
        fullpath = os.path.join(outputPath + "files/" + fnamestress)
        stressField.save(fullpath)
f_o.close()


# In[36]:

#fig1 = plt.Figure()
#fig1.Surface(buoyancyFn[1], elementMesh)
#fig1.Surface(viscosityl1, elementMesh)
#fig1.Points( swarm=gSwarm, colourVariable=viscVariable , pointSize=3.0)
#fig1.VectorArrows(velocityField, linearMesh, lengthScale=0.02)
#fig1.show()


# In[40]:

figTemp = glucifer.Figure()
figTemp + glucifer.objects.Surface(elementMesh, temperatureField)
figTemp + glucifer.objects.VectorArrows(elementMesh,velocityField, arrowHead=0.2, scaling=0.1)
figTemp+ glucifer.objects.Mesh(linearMesh)
figTemp.show()


# In[41]:

print(time.clock()-start, realtime)


# In[ ]:



