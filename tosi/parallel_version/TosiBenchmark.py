
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

# In[93]:

import underworld as uw
import math
from underworld import function as fn
import glucifer
#import matplotlib.pyplot as pyplot
import time
import numpy as np
import os
import shutil
import natsort

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


# Set physical constants and parameters, including the Rayleigh number (*RA*). 

# In[94]:

#Do you want to write hdf5 files - Temp, RMS, viscosity, stress?
writeFiles = False


# In[95]:

#ETA_T = 1e5
#newvisc = math.exp(math.log(ETA_T)*0.64)
#newvisc


# In[96]:

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


# In[97]:

CASE = 4 # select identifier of the testing case (1-5)


# In[98]:

###########
#Physical parameters
###########

RA  = 1e2        # Rayleigh number
TS  = 0          # surface temperature
TB  = 1          # bottom boundary temperature (melting point)
ETA_T = 1e5
ETA_Y = case_dict[CASE]['ETA_Y']
ETA0 = 1e-3
TMAX = 3.0
IMAX = 1000
YSTRESS = case_dict[CASE]['YSTRESS']


# Alternatively, use a reference viscosity identical to Crameri and Tackley, i.e. normalised at 0.64.
# In this case the Rayligh number would go to:

# Simulation parameters. Resolution in the horizontal (*Xres*) and vertical (*Yres*) directions.

# In[ ]:




# In[99]:

###########
#Mesh / simulation parameters
###########
dim = 2          # number of spatial dimensions
RES = 64


# Select which case of viscosity from Tosi et al (2015) to use. Adjust the yield stress to be =1 for cases 1-4, or between 3.0 and 5.0 (in increments of 0.1) in case 5.

# In[100]:

###########
#Model Runtime parameters
###########


files_output = 1e6
gldbs_output = 1000
images_output = 1e6
checkpoint_every = 200
metric_output = 20

comm.Barrier() #Barrier here so not procs run the check in the next cell too early 

assert metric_output <= checkpoint_every, 'Checkpointing should run less or as often as metric output'


# Set output file and directory for results

# In[101]:

###########
#Standard output directory setup
###########


outputPath = "results" + "/" +  str(CASE) + "/" + str(RES) + "/" 
imagePath = outputPath + 'images/'
filePath = outputPath + 'files/'
checkpointPath = outputPath + 'checkpoint/'
dbPath = outputPath + 'gldbs/'
outputFile = 'results_model' + str(CASE) + '_' + str(RES) +  '.dat'

if uw.rank()==0:
    # make directories if they don't exist
    if not os.path.isdir(outputPath):
        os.makedirs(outputPath)
    if not os.path.isdir(checkpointPath):
        os.makedirs(checkpointPath)
    if not os.path.isdir(imagePath):
        os.makedirs(imagePath)
    if not os.path.isdir(dbPath):
        os.makedirs(dbPath)
    if not os.path.isdir(filePath):
        os.makedirs(filePath)
        
comm.Barrier() #Barrier here so not procs run the check in the next cell too early 


# In[102]:

###########
#Check if starting from checkpoint
###########

checkdirs = []
for dirpath, dirnames, files in os.walk(checkpointPath):
    if files:
        print dirpath, 'has files'
        checkpointLoad = True
        checkdirs.append(dirpath)
    if not files:
        print dirpath, 'is empty'
        checkpointLoad = False
        


# In[ ]:




# Create mesh objects. These store the indices and spatial coordiates of the grid points on the mesh.

# In[103]:

mesh = uw.mesh.FeMesh_Cartesian( elementType = ("Q1/dQ0"),
                                 elementRes  = (RES, RES), 
                                 minCoord    = (0.,0.), 
                                 maxCoord=(1.,1.))


# Create Finite Element (FE) variables for the velocity, pressure and temperature fields. The last two of these are scalar fields needing only one value at each mesh point, while the velocity field contains a vector of *dim* dimensions at each mesh point.

# In[104]:

velocityField       = uw.mesh.MeshVariable( mesh=mesh,         nodeDofCount=dim )
pressureField       = uw.mesh.MeshVariable( mesh=mesh.subMesh, nodeDofCount=1 )
temperatureField    = uw.mesh.MeshVariable( mesh=mesh,         nodeDofCount=1 )
temperatureDotField = uw.mesh.MeshVariable( mesh=mesh,         nodeDofCount=1 )


# Create some dummy fevariables for doing top and bottom boundary calculations.

# In[105]:

topField    = uw.mesh.MeshVariable( mesh=mesh,   nodeDofCount=1)
bottomField    = uw.mesh.MeshVariable( mesh=mesh,   nodeDofCount=1)

topField.data[:] = 0.
bottomField.data[:] = 0.

# lets ensure temp boundaries are still what we want 
# on the boundaries
for index in mesh.specialSets["MinJ_VertexSet"]:
    bottomField.data[index] = 1.
for index in mesh.specialSets["MaxJ_VertexSet"]:
    topField.data[index] = 1.


# #ICs and BCs

# In[106]:

# Initialise data.. Note that we are also setting boundary conditions here
velocityField.data[:] = [0.,0.]
pressureField.data[:] = 0.
temperatureField.data[:] = 0.
temperatureDotField.data[:] = 0.


# Setup temperature initial condition via numpy arrays
A = 0.01
#Note that width = height = 1
tempNump = temperatureField.data
for index, coord in enumerate(mesh.data):
    pertCoeff = (1- coord[1]) + A*math.cos( math.pi * coord[0] ) * math.sin( math.pi * coord[1] )
    tempNump[index] = pertCoeff;
    


# In[107]:

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
IWalls = mesh.specialSets["MinI_VertexSet"] + mesh.specialSets["MaxI_VertexSet"]
JWalls = mesh.specialSets["MinJ_VertexSet"] + mesh.specialSets["MaxJ_VertexSet"]
TWalls = mesh.specialSets["MaxJ_VertexSet"]
BWalls = mesh.specialSets["MinJ_VertexSet"]


# In[108]:

# Now setup the dirichlet boundary condition
# Note that through this object, we are flagging to the system 
# that these nodes are to be considered as boundary conditions. 
# Also note that we provide a tuple of sets.. One for the Vx, one for Vy.
freeslipBC = uw.conditions.DirichletCondition(     variable=velocityField, 
                                              indexSetsPerDof=(IWalls, JWalls) )

# also set dirichlet for temp field
tempBC = uw.conditions.DirichletCondition(     variable=temperatureField, 
                                              indexSetsPerDof=(JWalls,) )


# In[109]:

# Set temp boundaries 
# on the boundaries
for index in mesh.specialSets["MinJ_VertexSet"]:
    temperatureField.data[index] = TB
for index in mesh.specialSets["MaxJ_VertexSet"]:
    temperatureField.data[index] = TS


# In[110]:

if checkpointLoad:
    checkpointLoadDir = natsort.natsorted(checkdirs)[-1]
    temperatureField.load(os.path.join(checkpointLoadDir, "temperatureField" + ".hdf5"))
    pressureField.load(os.path.join(checkpointLoadDir, "pressureField" + ".hdf5"))
    velocityField.load(os.path.join(checkpointLoadDir, "velocityField" + ".hdf5"))


# #Material properties
# 

# In[111]:

#Make variables required for plasticity

secinvCopy = fn.tensor.second_invariant( 
                            fn.tensor.symmetric( 
                            velocityField.fn_gradient ))


# In[112]:

coordinate = fn.input()


# In[113]:

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


# In[114]:

print(RA, ETA_T, ETA_Y, ETA0, YSTRESS)


# Set up simulation parameters and functions
# ====
# 
# Here the functions for density, viscosity etc. are set. These functions and/or values are preserved for the entire simulation time. 

# In[115]:

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

# In[116]:

#We first set up a l
stokesPIC = uw.systems.Stokes(velocityField=velocityField, 
                              pressureField=pressureField,
                              conditions=[freeslipBC,],
#                              viscosityFn=viscosityFn1, 
                              fn_viscosity=fn.exception.SafeMaths(viscosityFn1), 
                              fn_bodyforce=buoyancyFn)


# We do one solve with linear viscosity to get the initial strain rate invariant. This solve step also calculates a 'guess' of the the velocity field based on the linear system, which is used later in the non-linear solver.

# In[117]:

solver = uw.systems.Solver(stokesPIC)
solver.solve() 


# In[118]:

# Setup the Stokes system again, now with linear or nonlinear visocity viscosity.
stokesPIC2 = uw.systems.Stokes(velocityField=velocityField, 
                              pressureField=pressureField,
                              conditions=[freeslipBC,],
                              fn_viscosity=fn.exception.SafeMaths(viscosityFn2), 
                              fn_bodyforce=buoyancyFn )


# In[119]:

solver = uw.systems.Solver(stokesPIC2) # altered from PIC2


# Solve for initial pressure and velocity using a quick non-linear Picard iteration
# 

# In[120]:

solver.solve(nonLinearIterate=True)


# Create an advective-diffusive system
# =====
# 
# Setup the system in underworld by flagging the temperature and velocity field variables.

# In[121]:


advDiff = uw.systems.AdvectionDiffusion( phiField       = temperatureField, 
                                         phiDotField    = temperatureDotField, 
                                         velocityField  = velocityField, 
                                         fn_diffusivity = 1.0, 
                                         conditions     = [tempBC,] )


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

# In[122]:

#Setup some Integrals. We want these outside the main loop...
tempint = uw.utils.Integral(temperatureField, mesh)
areaint = uw.utils.Integral(1.,mesh)

v2int = uw.utils.Integral(fn.math.dot(velocityField,velocityField), mesh)
topareaint = uw.utils.Integral((topField*1.),mesh)

dwint = uw.utils.Integral(temperatureField*velocityField[1], mesh)

secinv = fn.tensor.second_invariant(
                    fn.tensor.symmetric(
                        velocityField.fn_gradient))

sinner = fn.math.dot(secinv,secinv)
vdint = uw.utils.Integral((4.*viscosityFn2*sinner), mesh)


# In[123]:

def avg_temp():
    return tempint.evaluate()[0]/areaint.evaluate()[0]


def nusseltNumber(temperatureField, temperatureMesh, indexSet):
    tempgradField = temperatureField.fn_gradient
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
    vuviscfn.evaluate(mesh)
    return vuviscfn.max_global(), vuviscfn.min_global()


# In[124]:

avg_temp()


# In[125]:

#Fields for saving data / fields

rmsField = uw.mesh.MeshVariable( mesh=mesh,   nodeDofCount=1)
rmsfn = fn.math.sqrt(fn.math.dot(velocityField,velocityField))
rmsdata = rmsfn.evaluate(mesh)
rmsField.data[:] = rmsdata 

viscField = uw.mesh.MeshVariable( mesh=mesh,   nodeDofCount=1)
viscdata = viscosityFn2.evaluate(mesh)
viscField.data[:] = viscdata


stressField = uw.mesh.MeshVariable( mesh=mesh,   nodeDofCount=1)
srtdata = fn.tensor.second_invariant( 
                    fn.tensor.symmetric( 
                        velocityField.fn_gradient ))

rostfield = srtdata.evaluate(mesh)
stressinv = 2*viscdata*rostfield[:]
stressField.data[:] = stressinv


# In[126]:

def checkpoint1(step, checkpointPath,filename, filewrites):
    path = checkpointPath + str(step) 
    os.mkdir(path)
    ##Write and save the file, if not already a writing step
    if not step % filewrites == 0:
        filename.write((13*'%-15s ' + '\n') % (realtime, Viscdis, float(Nu0glob), float(Nu1glob), Avg_temp, 
                                              Rms,Rmsurfglob,Max_vx_surf,Gravwork, etamax, etamin, Viscdisair, Viscdislith))
    filename.close()
    shutil.copyfile(os.path.join(outputPath, outputFile), os.path.join(path, outputFile))


def checkpoint2(step, checkpointPath,  filename):
    path = checkpointPath + str(step) 
    velfile = "velocityField" + ".hdf5"
    tempfile = "temperatureField" + ".hdf5"
    pressfile = "pressureField" + ".hdf5"
    velocityField.save(os.path.join(path, velfile))
    temperatureField.save(os.path.join(path, tempfile))
    pressureField.save(os.path.join(path, pressfile))
        


# Main simulation loop
# =======
# 
# The main time stepping loop begins here. Before this the time and timestep are initialised to zero and the output statistics arrays are set up. Also the frequency of outputting basic statistics to the screen is set in steps_output.
# 

# In[127]:

realtime = 0.
step = 0
timevals = [0.]
steps_end = 10
steps_output = 1e6
steps_display_info = 20


# In[128]:

###########
#Open file for writing metrics
###########

if checkpointLoad:
    if uw.rank() == 0:
        shutil.copyfile(os.path.join(checkpointLoadDir, outputFile), outputPath+outputFile)
    comm.Barrier()
    #os.rename(os.path.join(checkpointLoadDir, outputFile), outputPath+outputFile)
    f_o = open(os.path.join(outputPath, outputFile), 'a')
    prevdata = np.genfromtxt(os.path.join(outputPath, outputFile), skip_header=0, skip_footer=0)
    realtime = prevdata[prevdata.shape[0]-1, 0]
    step = int(checkpointLoadDir.split('/')[-1])
    timevals = [0.]
else:
    f_o = open(outputPath+outputFile, 'w')
    realtime = 0.
    step = 0
    timevals = [0.]


# In[129]:

step


# In[130]:

# initialise timer for computation
start = time.clock()
# Perform steps
while realtime < 0.15:
#while step < steps_end:
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
    ################            
    # Calculate the Metrics, only on 1 of the processors:
    ################
    if (step % metric_output == 0):
    
        Avg_temp = avg_temp()
        Rms = rms()
        Rms_surf = rmsBoundary(velocityField,mesh, TWalls)
        Max_vx_surf = max_vx_surf(velocityField, mesh)
        Gravwork = gravwork(dwint)
        Viscdis = viscdis(vdint)
        nu0, nu1 = nusseltNumber(temperatureField, mesh, BWalls), nusseltNumber(temperatureField, mesh, TWalls)
        etamax, etamin = visc_extr(viscosityFn2)
        # output to summary text file
        if uw.rank()==0:
            f_o.write((11*'%-15s ' + '\n') % (realtime, Viscdis, nu0, nu1, Avg_temp, Rms,Rms_surf,Max_vx_surf,Gravwork, etamax, etamin))
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
    ################
    #Checkpoint
    ################
    if step % checkpoint_every == 0:
        if uw.rank() == 0:
            checkpoint1(step, checkpointPath,f_o, metric_output)           
        checkpoint2(step, checkpointPath, f_o)
        f_o = open(os.path.join(outputPath, outputFile), 'a') #is this line supposed to be here?


# In[ ]:

f_o.close()


# In[ ]:

figTemp = glucifer.Figure()
figTemp.append( glucifer.objects.Surface(mesh, temperatureField))
#figTemp.append( glucifer.objects.Mesh(mesh))

figTemp.append( glucifer.objects.VectorArrows(mesh,velocityField, arrowHead=0.2, scaling=0.05))
#figTemp.save_database('test.gldb')
figTemp.show()


# In[41]:

print(time.clock()-start, realtime)


# In[ ]:



