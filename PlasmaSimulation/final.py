# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 05:28:19 2026

@author: Chris
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')

class PICSim:
    def __init__(self, Length=30, NumPoint=1024, NumPart=50000, dt=0.1):
        #System Dimensions (1D)
        self.Length = Length
        self.NumPoint = NumPoint
        self.NumPart = NumPart
        self.dt = dt
        self.dx = Length / NumPoint
        
        #Calculates k for FFT, these dont change
        self.k = 2 * np.pi * np.fft.fftfreq(NumPoint, d=self.dx) 
        self.k[0] = 1
        
        #Electron charge
        self.q = -1
        
        #Uniformly random starting position of particles
        self.x = np.random.uniform(0, Length, NumPart)
        
        #Create beams
        BeamV = 1.0
        ThermalV = 0.1
        self.Vel = np.concatenate(( np.random.normal(BeamV, ThermalV, NumPart//2), np.random.normal(-BeamV, ThermalV, NumPart//2))) 
        
        #Initial perturbation
        self.x += 0.1*np.cos(2*np.pi*self.x/Length)
        self.x = self.x % Length #Particles that move past the end wrap back around
        
        #sometihng    
        self.t_history = []
        self.energy_history = []
        self.time = 0.0
        
        #Leapfrog integration setup
        rho = self.ScatterCharge(self.x, NumPoint, self.dx, self.q)
        E = self.SolveField(rho, self.k)
        EPart = self.GatherCharge(self.x, NumPoint, E, self.dx)
        self.Vel -= 0.5 * self.dt * (-EPart) #q/m = 1
        

    def Step(self):
        #The PIC cycle
        rho = self.ScatterCharge(self.x, self.NumPoint, self.dx, self.q)      
        E = self.SolveField(rho, self.k)
        EPart = self.GatherCharge(self.x, self.NumPoint, E,self.dx)
        
        #Leapfrog particle movement
        self.Vel += self.dt * (-EPart)
        self.x += self.dt * self.Vel
        self.x = self.x % self.Length
        
        #somerhing
        self.time += self.dt
        field_energy = 0.5 * np.sum(E**2) * self.dx
        
        #Store for plotting
        self.t_history.append(self.time)
        self.energy_history.append(field_energy)
        return E
    
    def ScatterCharge(self, x, NumPoint, dx, q):
        NormX = x / dx #Position in Grid
        
        #Gives indices of grid points
        j = np.floor(NormX).astype(int) 
        jPlus = (j + 1) % NumPoint
        
        w = NormX - j #Weighting of charge
        
        rho = np.zeros(NumPoint) #Charge array
        
        #Add charge at each index
        np.add.at(rho, j, (1-w)*q) 
        np.add.at(rho, jPlus, w*q)
        
        #Add positive background for total neutral charge
        BackgroundDensity = -np.sum(rho) / NumPoint 
        rho += BackgroundDensity 
        return rho
    
    def SolveField(self, rho, k):
        rhoK = np.fft.fft(rho) #Charge density in k-space
                      
        EK = -1j * rhoK / k #E Field in k-space
        
        EK[0] = 0 #E at k=0 would be a constant background field, so must be set to 0
        
        E = np.fft.ifft(EK) #E in traditional x-space  
        
        return E.real #E occasionally has a small imaginary part
    
    def GatherCharge(self, x, NumPoint, E, dx):
        NormX = x / dx #Position in Grid
        
        #Gives indices of grid points
        j = np.floor(NormX).astype(int) 
        jPlus = (j + 1) % NumPoint
        
        w = NormX - j #Weighting of E
        
        EPart = E[j]*(1-w) + E[jPlus]*w #E at particle position     
        return EPart
        
    def run(self, steps):
        plt.ion() 
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})
        plt.tight_layout(pad=4.0)
        
        # Force the window to physically appear before the loop starts
        fig.show() 

        for t in range(steps):
            self.Step() 
            
            self.plot(ax1, ax2, t)
            if t % 10 == 0:  
                
                
                # --- The Magic Redraw Commands ---
                fig.canvas.draw()
                fig.canvas.flush_events() 
                
                print(f"Simulation is running... Step {t}")
                
        plt.ioff() 
        plt.show(block=True)
        
    def plot(self, ax1, ax2, t):
        # --- TOP PLOT: Phase Space ---
        ax1.clear()
        half = self.NumPart // 2
        ax1.scatter(self.x[:half], self.Vel[:half], s=0.2, color='blue', edgecolors='none')
        ax1.scatter(self.x[half:], self.Vel[half:], s=0.2, color='red', edgecolors='none')   
        ax1.set_xlim(0, self.Length)
        ax1.set_ylim(-5, 5)
        ax1.set_xlabel('Position')
        ax1.set_ylabel('Velocity')
        ax1.set_title(f'Phase Space (Step: {t})')

        # --- BOTTOM PLOT: Field Energy ---
        ax2.clear()
        if len(self.energy_history) > 0:
            ax2.plot(self.t_history, self.energy_history, color='black', linewidth=1.5)
            ax2.set_yscale('log') 
            
        ax2.set_xlim(0, max(15, self.time))
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Field Energy ($\Sigma E^2$)')
        ax2.set_title('Electric Field Energy')
        ax2.grid(True, which="both", linestyle="--", alpha=0.5)
   
        
   
NewSim = PICSim()
NewSim.run(1000)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    