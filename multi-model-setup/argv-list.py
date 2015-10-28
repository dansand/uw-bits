import sys
import os
import math

print('sys.argv is', sys.argv)

#if sys.argv[3]:
#    print('sys.argv 3 is', sys.argv[3])

Model = "A" # select identifier of the testing case (1-5)
ModNum = str(0)
if len(sys.argv) > 1:
    ModIt = str(sys.argv[1])
else:
    ModIt = "Base"

###########
#Constants
###########
#These models use a reference viscosity normalised at T = 0.5, z = 1.; rather than T = 0. in Tosi
#Crameri and Tckley use 0.64, because internal heating applies.
ETA_T = 1e5
refvisc = math.exp(math.log(ETA_T)*0.64)
#RA  = 1e2*refvisc       # Rayleigh number
RA  = 1.5e7       # Rayleigh number
TS  = 0          # surface temperature
TB  = 1          # bottom boundary temperature (melting point)
ETA_Y = 10.
ETA0 = 1e-3*refvisc
MAXY = 1.2
RES = 128

##########
#variables, these can be iterated over:
##########
if len(sys.argv) > 1:
    YSTRESS = float(sys.argv[1])*refvisc
else:
    YSTRESS = 1.*refvisc

outputPath = 'Output/' + str(Model) + "/" + str(ModNum) + "/"
imagePath = outputPath + 'images/'
filePath = outputPath + 'files/'
dbPath = outputPath + 'gldbs/'
outputFile = 'results_model' + Model + ModNum + ModIt + '.dat'

# make directories if they don't exist
if not os.path.isdir(outputPath):
    os.makedirs(outputPath)
if not os.path.isdir(imagePath):
    os.makedirs(imagePath)
if not os.path.isdir(dbPath):
    os.makedirs(dbPath)
if not os.path.isdir(filePath):
    os.makedirs(filePath)
