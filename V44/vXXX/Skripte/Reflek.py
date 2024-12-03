import matplotlib.pyplot as plt
import numpy as np
from uncertainties import ufloat
import uncertainties.unumpy as unp
import pandas as pd
import scipy.constants as const
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import math
import plot

alpha=0.4584
D=20e-3
d0=0.16e-3
Imax=3.664e5
Lambda=1.54e-10
Reflekt = plot.Data("/mnt/c/Users/mas19/OneDrive/Uni/1.Semester/FP-Master/Protokolle/FP/V44/vXXX/Daten/Messung1.UXD", 56)
Diffus = plot.Data("/mnt/c/Users/mas19/OneDrive/Uni/1.Semester/FP-Master/Protokolle/FP/V44/vXXX/Daten/Messung2.UXD", 56)
RefwoD=np.array([Reflekt[:,0],Reflekt[:,1]-Diffus[:,1]]).transpose()
Reflekt[:,1]=Reflekt[:,1]/(Imax*5)
Diffus[:,1]=Diffus[:,1]/(Imax*5)
RefwoD[:,1]=RefwoD[:,1]/(Imax*5)

def PlotRef(Reflekt,Difus,RefwoD):
    fig, ax = plot.Plot7(Reflekt[:,0], Reflekt[:,1], r"$\Theta$ / °", r"Reflexionsgrad R","gemessene Reflektivität")
    Difus=Difus[Difus[:,1] != 0]
    ax.plot(Difus[:,0], Difus[:,1],"y-", label="Diffuse Streuung")
    ax.plot(RefwoD[:,0], RefwoD[:,1],color='green', linestyle='-', label="Reflektivität ohne Diffuse Streuung",linewidth=1)
    ax.set_yscale('log')
    plt.grid()
    plt.legend(loc="upper right")
    plt.savefig("/mnt/c/Users/mas19/OneDrive/Uni/1.Semester/FP-Master/Protokolle/FP/V44/vXXX/plots/Reflektionsmessung.pdf")
    plt.clf()
#PlotRef(Reflekt,Diffus,RefwoD)
GFaktor = np.where(Reflekt[:, 0] < alpha, D * np.sin(Reflekt[:, 0] * np.pi / 180) / d0, 1)
KRefwoD = np.array([RefwoD[1:, 0], RefwoD[1:, 1] / GFaktor[1:]]).transpose()
alphaSi=0.223
t=np.sqrt(Reflekt[1:,0]**2-alphaSi**2)
r=(Reflekt[1:,0]-t)/(Reflekt[1:,0]+t)
RFresenel=np.where(Reflekt[1:,0]>alphaSi,r**2,1)
peaks, properties = find_peaks(-np.log(KRefwoD[:350,1]), height=(1,10.4), distance=6,prominence=2e-5)
Deltaalpha=np.mean(np.diff(KRefwoD[peaks,0]))
print("Deltaalpha=",Deltaalpha)
Schichtdicke=180*Lambda/(2*Deltaalpha*np.pi)
print("Schichtdicke=",Schichtdicke)

def PlotKorr(RefwoD,KRefwoD,RFresenel,peaks):
    fig, ax = plot.Plot7(RefwoD[:,0], RefwoD[:,1], r"$\Theta$ / °", r"Reflektivität R","Reflektivität ohne Diffuse Streuung")
    ax.plot(KRefwoD[:,0], KRefwoD[:,1],color='green', linestyle='-', label=r"Reflektivität $/G$",linewidth=1)
    ax.plot(Reflekt[1:,0], RFresenel[:],"y-", label="Fresnelreflektivität für Silizium")
    ax.plot(KRefwoD[peaks,0], KRefwoD[peaks,1],"kx", label="Minima der Kiessig-Oszillationen")
    ax.axvline(x=0.223, color='lightblue', linestyle='--',label="Kritischer Winkel für Silizium")
    ax.axvline(x=0.152, color='lightblue', linestyle='--',label="Kritischer Winkel für Polysterol")
    ax.set_yscale('log')
    plt.grid()
    plt.legend(loc="upper right")
    plt.savefig("/mnt/c/Users/mas19/OneDrive/Uni/1.Semester/FP-Master/Protokolle/FP/V44/vXXX/plots/KorrigierteReflektionsmessung.pdf")
    plt.clf()
#PlotKorr(RefwoD,KRefwoD,RFresenel,peaks)

DeltaSi=7.1e-6
DeltaPol=3.5e-6
sigmaSi=11e-10
sigmaPol=15e-10
beta1=6e-7
beta2=0.1e-6
z2=8.1e-8
print("DeltaSi=",DeltaSi,"DeltaPol=",DeltaPol,"sigmaSi=",sigmaSi,"sigmaPol=",sigmaPol,"beta1=",beta1,"beta2=",beta2,"z2=",z2)
def ParattRau(DeltaSi,DeltaPol,sigmaSi,sigmaPol,beta1,beta2,z2,a_i):
    a_i=a_i*np.pi/180
    f1=np.sin(a_i)
    f2=np.sqrt(np.sin(a_i)**2-2j*beta1-2*DeltaSi) #Näherung der Wurzel n^2-cos^2
    f3=np.sqrt(np.sin(a_i)**2-2j*beta2-2*DeltaPol)
    k_1=2*np.pi/Lambda*f1
    k_2=2*np.pi/Lambda*f2
    k_3=2*np.pi/Lambda*f3
    r12=(k_1-k_2)/(k_1+k_2)*np.exp(-2*k_1*k_2*sigmaSi**2)
    r23=(k_2-k_3)/(k_2+k_3)*np.exp(-2*k_2*k_3*sigmaPol**2)

    X2=np.exp(-2j*k_2*z2)*r23
    X1=(r12+X2)/(1+r12*X2)
    return abs(X1)**2
    
R=ParattRau(DeltaSi,DeltaPol,sigmaSi,sigmaPol,beta1,beta2,z2,Reflekt[:,0])

def Plotparrat(KRefwoD,R):
    fig, ax = plot.Plot7(KRefwoD[:,0], KRefwoD[:,1], r"$\Theta$ / °", r"Reflektivität $R/G$","Reflektivität/Geometriefaktor")
    ax.set_yscale('log')
    ax.plot(KRefwoD[:,0], R[1:],"y-", label="Theoriekurve mit Paratt-Algorithmus")
    ax.set_ylim(1e-6,1e1)
    plt.grid()
    plt.legend(loc="upper right")
    plt.savefig("/mnt/c/Users/mas19/OneDrive/Uni/1.Semester/FP-Master/Protokolle/FP/V44/vXXX/plots/Parattplot.pdf")
    plt.clf()
Plotparrat(KRefwoD,R)