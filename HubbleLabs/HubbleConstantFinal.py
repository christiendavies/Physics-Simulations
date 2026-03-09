import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

def Main():
    Alpha, Beta, AlphaError, BetaError = FindAlphaBeta()
    NGCDist, NGCDistError = FindGalDistance(Alpha, Beta, AlphaError, BetaError)
    HubbleConstant, HubbleConstError = FindHubbleConst(NGCDist, NGCDistError)
    
    print("Alpha=",Alpha) 
    print("AlphaError=",AlphaError)
    print("Beta=",Beta)
    print("BetaError=",BetaError)
    print("NGC4527 Distance=",NGCDist)
    print("NGC Distance Error=",NGCDistError)
    print("HubbleConstant=",HubbleConstant)
    print("HubbleC Error=",HubbleConstError)
    
def ChiSquared(DataY, ModelY, ErrorY):
    """
    Function to calculate the Chi squared value given a dataset
    Inputs: DataY - Dataset obtained from the actual values provided
            ModelY - Dataset obtained from the model
            ErrorY - List of errors for each value in DataY
    Outputs: Chi2 - The calculated value of Chi squared
    """
    DataY = np.array(DataY) #ensures all input data is in the correct format
    ModelY = np.array(ModelY)
    ErrorY = np.array(ErrorY)  
    Chi2 = np.sum(((DataY-ModelY)**2)/(ErrorY**2)) #formula for Chi squared
    return Chi2

def AbsMagFormula(m, d, A):
    return m - 5*np.log10(d) + 5 - A #formula for absolute magnitude

def LinearRelation(x, slope, intercept): #formula used for fitting Luminosity and Period
    return slope*x + intercept

def LinearRelationNoIntercept(x, slope): #formula used for fitting the Hubble constant
    return slope*x


def FindAlphaBeta():
    """
    Function to calculate optimal values of Alpha and Beta with errors in the Period-Luminosity Relationship
    Outputs: Alpha - Float of the optimal value for the slope of the fit
             Beta - Float of the optimal value for intercept of the fit
             AlphaError, BetaError - Floats indicating the uncertainty on both these values
    """
    Parallax, ParErr, Period, AppMag, Extinction, ExtErr = np.loadtxt('MW_Cepheids.dat', unpack=True, usecols=(1,2,3,4,5,6))
    NumDataPoints = len(Parallax)
    Distance = 1000/Parallax
    AbsMag = AbsMagFormula(AppMag, Distance, Extinction) #creates list of same length as input data of absolute magnitudes
    LogPeriod = np.log10(Period)
    DoF = NumDataPoints - 2


    CalcMagMean = [0]*NumDataPoints
    CalcMagError = [0]*NumDataPoints    
    for i in range(NumDataPoints):
        AppMagError = np.random.normal(AppMag[i], 0, size=10000) #needed to combine errors with known values
        ParallaxError = np.random.normal(Parallax[i], ParErr[i], size=10000) #gaussian distributions for each datapoint
        ExtinctionError = np.random.normal(Extinction[i], ExtErr[i], size=10000)
        
        TestAbsMag = AbsMagFormula(AppMagError, 1000/ParallaxError, ExtinctionError)#gaussian of each result of absolute magnitude
        CalcMagMean[i] = np.mean(TestAbsMag)
        CalcMagError[i] = np.std(TestAbsMag)
        
      
    InitialParameters, ParametersCov = opt.curve_fit(f=LinearRelation, xdata=LogPeriod, ydata=CalcMagMean, sigma=CalcMagError, absolute_sigma=True)
    Alpha = InitialParameters[0] #obtains the optimal parameters from the curve_fit
    Beta = InitialParameters[1]
    AlphaError = np.sqrt(ParametersCov[0,0])#obtains the errors on the parameters from the covariance matrix of curve_fit
    BetaError = np.sqrt(ParametersCov[1,1])
    

    plt.scatter(LogPeriod,AbsMag)
    plt.xlabel("LogPeriod")
    plt.ylabel("AbsoluteMag")
    plt.title("Period-Luminosity")
    plt.plot(LogPeriod, LinearRelation(LogPeriod, Alpha, Beta))
    plt.errorbar(LogPeriod, AbsMag, yerr=CalcMagError, fmt='o', capsize=5)
    plt.show()         
    
    BestY = LinearRelation(LogPeriod, Alpha, Beta)#calculates what the model predicts absolute magnitude as at each value of LogPeriod
    Chi2 = ChiSquared(CalcMagMean, BestY, CalcMagError) #chi squared and reduced chi squared from the model
    RedChi2 = Chi2/DoF
    print("Period Luminosity Chi2=",Chi2)
    print("Period Luminosity RedChi2=",RedChi2)
    return Alpha, Beta, AlphaError, BetaError


def FindGalDistance(Alpha, Beta, AlphaError, BetaError):
    """
    Function to find the distance of a galaxy from a set of stars in the galaxy
    Inputs: Alpha, Beta - Floats giving the optimal values for the period-luminosity relationship
            AlphaError, BetaError - Floats giving the errors for each optimal value
    Outputs: NGCDistance - The optimal value of distance to this galaxy
             NGCDistError - The associated error with the estimate of distance
    """
    LogPeriod, AppMag = np.loadtxt("ngc4527_cepheids.dat", unpack=True, usecols=(1,2))
    NumDataPoints = len(LogPeriod)
    NGCExtinction = 0.0682
    
    CalcDistanceMean = [0]*NumDataPoints
    CalcDistanceError = [0]*NumDataPoints
    for i in range(NumDataPoints):       
        EquationAlpha = np.random.normal(Alpha, AlphaError, size=10000) #gaussians of the fit parameters for each datapoint
        EquationBeta = np.random.normal(Beta, BetaError, size=10000)
    
        CalcDistance = 10**((AppMag[i] + 5 - NGCExtinction - EquationAlpha*LogPeriod[i] - EquationBeta)/5)#gaussian of distance for each datapoint
        CalcDistanceMean[i] = np.mean(CalcDistance) 
        CalcDistanceError[i] = np.std(CalcDistance)
        
    NGCDistance = np.average(CalcDistanceMean, weights=CalcDistanceError) #finds distance using weighted mean
    DistVariance = np.average((CalcDistanceMean - NGCDistance)**2, weights=CalcDistanceError)
    NGCDistError = np.sqrt(DistVariance / np.sum(CalcDistanceError)) #finds the error on weighted mean
    return NGCDistance, NGCDistError


def FindHubbleConst(NGCDist, NGCDistError):
    """
    Function to find the optimal value for the Hubble constant
    Inputs: NGCDist, NGCDistError - floats to be added to the dataset given 
    Outputs: HubbleConstant - float of the optimal estimate of H0
            HubbleConstantError - float of the estimated error on this value of H0
    """
    RecVel, GalDistance, GalDistError = np.loadtxt('other_galaxies.dat', unpack=True, usecols=(1,2,3))   
    RecVel = np.append(RecVel, 1152) #adds values from step 2 into the new dataset
    GalDistance = np.append(GalDistance, NGCDist/(10**6))
    GalDistError = np.append(GalDistError, NGCDistError/(10**6))
    DoF = len(RecVel) - 1
    
    InitialH = NGCDist/1152
    HubbleParameters, HubbleParameterCov = opt.curve_fit(f=LinearRelationNoIntercept, xdata=RecVel, ydata=GalDistance, sigma=GalDistError, p0=InitialH, absolute_sigma=True)
    HubbleConstant = 1/HubbleParameters[0] #is 1/ as im plotting distance against velocity
    HubbleConstantError = np.sqrt(HubbleParameterCov[0, 0])
    ErrorPercent = (HubbleConstantError/HubbleParameters[0])*100
    HubbleConstantError = HubbleConstant*ErrorPercent
    
    plt.scatter(GalDistance, RecVel)
    plt.xlabel("Galaxy Distance (Mpc)")
    plt.ylabel("Recession Velocity (km/s)")
    plt.title("Plot of Hubble's Law")
    plt.plot(GalDistance, LinearRelationNoIntercept(GalDistance, HubbleConstant))
    plt.errorbar(GalDistance, RecVel, xerr=GalDistError, fmt='o', capsize=5)
    plt.show()
    
    BestY = LinearRelationNoIntercept(GalDistance, HubbleConstant)
    Chi2 = ChiSquared(RecVel, BestY, HubbleConstantError) #chi squared and reduced chi squared from the model
    RedChi2 = Chi2/DoF
    print("Hubble Fit Chi2=", Chi2)
    print("Hubble Fit RedChi2=", RedChi2)
    return HubbleConstant, HubbleConstantError

Main()