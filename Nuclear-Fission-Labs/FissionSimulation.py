import numpy as np
from math import cos, sin, log, pi, sqrt, acos
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

class SimulationVolume:
    '''
    Holds the list of neutrons and runs the simulation
    '''
    def __init__(self, Volume, InitFissNum, Shape, Reflection=0.0):
        '''
        Creates the list of neutrons in the simulation
        
        Volume - array of side lengths of cube
        InitFissNum - number of initial fissions
        Shape - deciding to calculate for sphere or cube
        Reflection - the reflectivity of the boundary of the sphere, set to 0 if not given
        '''
        
        self.Shape = Shape
        self.NeList = []
        self.Count = 0 
        self.Reflection = Reflection
        
        if self.Shape == 'cube':
            self.XLength = Volume[0]
            self.YLength = Volume[1]
            self.ZLength = Volume[2]
                 
            for i in range(InitFissNum):
                XPos = np.random.uniform(0, self.XLength) #Generating a random position for a fission event to occur at
                YPos = np.random.uniform(0, self.YLength)
                ZPos = np.random.uniform(0, self.ZLength)
                Position = [XPos, YPos, ZPos]
                self.Fission(Position) #Creates a new fission passing the random position
        elif self.Shape == 'sphere':
            self.Radius = Volume[0]
            
            InsideCount = 0
            while InsideCount < InitFissNum:           
                XPos = np.random.uniform(0, self.Radius) #Generating a random position for a fission event to occur at
                YPos = np.random.uniform(0, self.Radius)
                ZPos = np.random.uniform(0, self.Radius)     
                
                if (XPos**2 + YPos**2 + ZPos**2) <= self.Radius**2:
                    Position = [XPos, YPos, ZPos]
                    self.Fission(Position)
                    InsideCount += 1
                
    def Fission(self, Position):
        '''
        Creates new neutrons at the specified fission position
        
        Position - neutron creation point 
        '''
        XPos = Position[0]
        YPos = Position[1]
        ZPos = Position[2]
        NePosition = [XPos, YPos, ZPos]
        for j in range(self.neutrons()): 
            if (Position[1] == 0) and (Position[2] == 0): #1D Case
                Direction = [np.random.random()]
                self.NeList.append(Neutron(NePosition, Direction))
            else: #3D Case
                theta = acos(2*np.random.random()-1) #Generating random direction for each neutron to move in
                phi = 2*pi*np.random.random()
                Direction = [theta, phi]
                self.NeList.append(Neutron(NePosition, Direction)) #Adding each new neutron to the list
        
                
    def RunSim(self, OrderNum):
        '''
        Starts the simulation and returns the total number of fissions 
        
        OrderNum - Number of steps the simulation is run for
        '''
        for i in range(OrderNum): 
            self.MoveNeutrons()
        
        return self.Count           
                
    def neutrons(self):
        """Number of secondary neutrons produced in each fission.

        Returns an integer number of neutrons, with average 2.5."""
        i=int(np.random.normal()+3.0)
        if (i<0): return 0
        else: return i
           
    def MoveNeutrons(self):
        '''
        Moves each neutron in the simulation and tests whether new position is inside the volume.
        In the sphere case, a neutron can move outside the volume, but be reflected back inside and counted
        '''
        CopyNeList = self.NeList[:] #Creates a copy of the list to iterate over
        for Ne in CopyNeList:
            Ne.Move()
            NePos = Ne.GetPos()
            
            if self.Shape == 'cube':            
                if (0 <= NePos[0] <= self.XLength) and (0 <= NePos[1] <= self.YLength) and (0 <= NePos[2] <= self.ZLength):
                    self.Count += 1 #Neutron starts another fission process so adds to the count
                    self.Fission(NePos) #Starts a new fission process at the same position as the neutron
                    self.NeList.remove(Ne) #Neutron has been absorbed
                else:
                    self.NeList.remove(Ne) #Neutron is outside the volume so is removed
                    
            elif self.Shape == 'sphere':
                TestR = NePos[0]**2 + NePos[1]**2 + NePos[2]**2
                
                if TestR <= self.Radius**2: #Inside sphere
                    self.Count += 1 
                    self.Fission(NePos)
                    self.NeList.remove(Ne)
                else: #Outside Sphere
                    if np.random.random() < self.Reflection: #Chance for neutron to be reflected
                        Ne.Reflect() #Reversing direction
                        Ne.Move() #Moving the neutron in the reverse direction
                        
                        NePos = Ne.GetPos() #Checking the reflected neutron lands back in the sphere
                        TestR = NePos[0]**2 + NePos[1]**2 + NePos[2]**2
                        if TestR <= self.Radius**2: #Reflected and travels back in sphere so causes a fission reaction.
                            self.Count += 1
                            self.Fission(Ne.GetPos())
                            self.NeList.remove(Ne)
                        else:
                            self.NeList.remove(Ne) #Reflected but not travelled enough to return to sphere
                    else:                
                        self.NeList.remove(Ne)
      
                
class Neutron:
    '''
    Holds the position and direction of one neutron
    '''
    def __init__(self, Position, Direction):
        '''
        Position - The position a new neutron is assigned
        Direction - A direction in spherical coordinates the neutron is emitted in
        '''
        self.XPos = Position[0]
        self.YPos = Position[1]
        self.ZPos = Position[2]
        if (self.YPos == 0) and (self.ZPos == 0): #1D Case
            self.Direction = Direction[0]
        else: #3D Case 
            self.Theta = Direction[0]
            self.Phi = Direction[1]
        
    def Move(self):
        '''
        Moves the neutron according to the assigned direction
        '''
        R = sqrt(2*0.017*0.21) 
   
        if (self.YPos == 0) and (self.ZPos == 0): #1D Case
            if self.Direction > 0.5: #Positive x direction
                self.XPos += R
            else: #Negative x direction
                self.XPos += -R
        else: #3D Case
            S = self.diffusion()
    
            XDist = R*S*sin(self.Theta)*cos(self.Phi) #Calculating the distance travelled in one step in each dimension
            YDist = R*S*sin(self.Theta)*sin(self.Phi)
            ZDist = R*S*cos(self.Theta)
          
            self.XPos += XDist #Moving positions by calculated distances
            self.YPos += YDist
            self.ZPos += ZDist
        
    
    def GetPos(self):
        return [self.XPos, self.YPos, self.ZPos]
        
    def Reflect(self):
        '''
        Reverses direction of neutron
        '''        
        self.Theta = np.pi - self.Theta
        self.Phi = np.pi + self.Phi
        
        
    def diffusion(self):
        """Distance diffused by a neutron before causing fission.

        Returns a random number with probability density p(s) =
        s^2 exp(-3s^2/R^2). This distribution has a mean of 1, so
        multiply by R to get the physical distance."""
        a=cos(2.0*pi*np.random.random())
        return sqrt(-0.667*(log(np.random.random())+log(np.random.random())*a*a))






###
'Q5.3'
###

def linear(x, m, c): #Linear function to fit to
    return m*x + c

def invlinear(y, m, c): #Inverse linear function to find length
    return (y-c)/m

InitFissions = 100
TestDimension = np.linspace(0.1, 0.15, 400)
Counts = []
CountsSTD = []
NumRuns = 10
for i in range(len(TestDimension)):
    RunCounts = []
    for j in range(NumRuns):
        Volume = SimulationVolume([TestDimension[i], 0, 0], InitFissions, 'cube')
        RunCounts.append(Volume.RunSim(1))
        
    Counts.append(np.mean(RunCounts))
    CountsSTD.append(np.std(RunCounts))   

popt, pcov = curve_fit(linear, TestDimension, Counts, sigma=CountsSTD)
CritLength = invlinear(InitFissions, popt[0], popt[1])
CritLengthError = CritLength*np.sqrt((np.sqrt(pcov[0,0])/popt[0])**2 + (np.sqrt(pcov[1, 1])/popt[1])**2)
Y = linear(TestDimension, popt[0], popt[1])
plt.plot(TestDimension, Counts)
plt.plot(TestDimension, Y, label='Linear Fit')
plt.xlabel('L (m)')
plt.ylabel('No. Secondary Fissions')
plt.title('Secondary fission counts against L in 1D')
plt.grid()
plt.legend()
plt.show()

print('Crit Length in 1D = ' + str(CritLength))
print('Crit Length Error in 1D = ' + str(CritLengthError))







###
'Q5.5'    
###
    
InitFissions = 100
TestDimension = np.linspace(0.12, 0.18, 400) #Range of L to test over
Counts = [] #Empty array to append mean values of counts
CountsSTD = [] #Empty array for std of counts at each L
NumRuns = 10 #Number of times to run the simulation at each L

for i in range(len(TestDimension)):
    RunCounts = [] #Empty array to append each value of the simulation to
    for j in range(NumRuns):    
        Volume = SimulationVolume([TestDimension[i], TestDimension[i], TestDimension[i]], InitFissions, 'cube')
        RunCounts.append(Volume.RunSim(1)) #Running simulation with one step to calculate secondary fissions
    
    Counts.append(np.mean(RunCounts))
    CountsSTD.append(np.std(RunCounts))  

popt, pcov = curve_fit(linear, TestDimension, Counts, sigma=CountsSTD) 
CritLength = invlinear(InitFissions, popt[0], popt[1])

CritLengthError = (1/popt[0])*np.sqrt((CritLength**2)*pcov[0,0] + pcov[1, 1])

print('Cube Critical L = ' + str(CritLength))
print('Cube Critical L Error = ' + str(CritLengthError))

CubeVolume = CritLength**3
Mass = CubeVolume * 18700 #18700 is the density of uranium
MassError = Mass * np.sqrt(3) * (CritLengthError/CritLength)

print('Cube Critical Mass = ' + str(Mass))
print('Cube Critical Mass Error = ' + str(MassError))

plt.plot(TestDimension, Counts)
Y = linear(TestDimension, popt[0], popt[1])
plt.plot(TestDimension, Y, color='red')
plt.xlabel('Side Length of Cube (m)')
plt.ylabel('No. Secondary Fissions')
plt.axhline(InitFissions, color='orange')
plt.axvline(CritLength, color='orange')
plt.title('Secondary fission counts over small L range')
plt.grid()
plt.show()







###
'Reduced ChiSq test for straight line fit'
###
Residuals = Counts - Y
ChiSq = np.sum((Residuals / CountsSTD)**2)
RedChiSq = ChiSq/398

print('Reduced Chi Squared of linear fit = ' + str(RedChiSq))






###
'Q5.5 Validity of linear fit'
###

WideTestDimension = np.linspace(0.1, 0.5, 1000)
Counts = []
for i in range(len(WideTestDimension)):
    Volume = SimulationVolume([WideTestDimension[i], WideTestDimension[i], WideTestDimension[i]], InitFissions, 'cube')
    Counts.append(Volume.RunSim(1))
    
plt.plot(WideTestDimension, Counts)
plt.plot(TestDimension, Y, color='red', label='Linear Fit')
plt.xlabel('Side Length of Cube (m)')
plt.ylabel('No. Secondary Fissions')
plt.legend()
plt.title('Secondary fission counts over large L range')
plt.grid()
plt.show()
   




     
###
'Q5.5 Validity of initial fission number'
###
   
RangeInitFission = np.linspace(10, 1000, 1000)
Counts = []
for num in RangeInitFission:
    Volume = SimulationVolume([0.15, 0.15, 0.15], int(num), 'cube')
    Counts.append(Volume.RunSim(1))
    
popt, pcov = curve_fit(linear, RangeInitFission, Counts)
Y = linear(RangeInitFission, popt[0], popt[1])
plt.plot(RangeInitFission, Counts)
plt.plot(RangeInitFission, Y, color='red', label = 'Linear Fit')
plt.xlabel('Initial Fission Number')
plt.ylabel('No. Secondary Fissions')
plt.grid()
plt.title('Secondary fission counts over range of initial fission numbers')
plt.legend()
plt.show()
     




####
'Extension for sphere with reflectivity'
####

InitFissions = 100
TestDimension = np.linspace(0.06, 0.12, 400)

Counts = [] 
CountsSTD = [] 
NumRuns = 10

for i in range(len(TestDimension)):
    RunCounts = [] 
    for j in range(NumRuns):    
        Volume = SimulationVolume([TestDimension[i]], InitFissions, 'sphere', Reflection=0.2)
        RunCounts.append(Volume.RunSim(1))
    
    Counts.append(np.mean(RunCounts))
    CountsSTD.append(np.std(RunCounts))  
    
popt, pcov = curve_fit(linear, TestDimension, Counts, sigma=CountsSTD) 
CritLength = invlinear(InitFissions, popt[0], popt[1])

CritLengthError = (1/popt[0])*np.sqrt((CritLength**2)*pcov[0,0] + pcov[1, 1])

print('Sphere Critical Radius = ' + str(CritLength))
print('Sphere Critical Radius Error = ' + str(CritLengthError))

CubeVolume = (4/3)*np.pi*(CritLength**3)
Mass = CubeVolume * 18700 #18700 is the density of uranium
MassError = Mass * 3 * (CritLengthError/CritLength)

print('Sphere Critical Mass = ' + str(Mass))
print('Sphere Critical Mass Error = ' + str(MassError))

Counts = [] 
NumRuns = 10
TestReflect = [0.1, 0.2, 0.3, 0.4]

for R in TestReflect:
    RunCounts = [] 
    for j in range(NumRuns):    
        Volume = SimulationVolume([0.08], InitFissions, 'sphere', Reflection=R)
        RunCounts.append(Volume.RunSim(1))
    
    Counts.append(np.mean(RunCounts))
    
plt.plot(TestReflect, Counts, marker='o', markersize=5)
plt.grid()
plt.title('Secondary counts over reflectivity range')
plt.xlabel('Reflectivity')
plt.ylabel('No. Secondary Fissions')
plt.show()
