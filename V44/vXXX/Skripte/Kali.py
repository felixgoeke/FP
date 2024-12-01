import matplotlib.pyplot as plt
import numpy as np
from uncertainties import ufloat
import uncertainties.unumpy as unp
import pandas as pd
import scipy.constants as const
from scipy.optimize import curve_fit
import math
import plot

xscan = plot.Data("/mnt/c/Users/mas19/OneDrive/Uni/1.Semester/FP-Master/Protokolle/FP/V44/vXXX/Daten/Xscan.UXD", 56)
Zscan = plot.Data("/mnt/c/Users/mas19/OneDrive/Uni/1.Semester/FP-Master/Protokolle/FP/V44/vXXX/Daten/Zscan.UXD", 56)
Rocking = plot.Data("/mnt/c/Users/mas19/OneDrive/Uni/1.Semester/FP-Master/Protokolle/FP/V44/vXXX/Daten/Rockingscan0.UXD", 56)
Detector= plot.Data("/mnt/c/Users/mas19/OneDrive/Uni/1.Semester/FP-Master/Protokolle/FP/V44/vXXX/Daten/Detektorscan.UXD", 56)

def gauss(x, A, mu, sigma,b):
    return A/np.sqrt(2*np.pi)/sigma * np.exp(-(x-mu)**2/(2*sigma**2))+b

def Detectorf(Detector):
    x_data = Detector[:, 0]
    y_data = Detector[:, 1]


    popt, pcov = curve_fit(gauss, x_data, y_data, p0=[max(y_data), np.mean(x_data), np.std(x_data),0])


    A_fit, mu_fit, sigma_fit, b_fit = popt
    x=np.linspace(-0.5,0.5,1000)
    print("A=",A_fit,"mu=",mu_fit,"sigma=",sigma_fit,"b=",b_fit)
    FWHM = 2 * np.sqrt(2 * np.log(2)) * sigma_fit
    x1=np.linspace(mu_fit-FWHM/2,mu_fit+FWHM/2,1000)

    fig,ax=plot.Plot6(Detector[:,0], Detector[:,1], r"$\Theta$ / °", r"Intensität $I[1/s]$","Messdaten")
    ax.plot(x, gauss(x, *popt), 'k-', label="Ausgleichkurve")
    ax.plot(x1,max(y_data)/2*np.ones(1000), color='y', linestyle='--', label="Halbwertsbreite")
    ax.axvline(x=mu_fit - FWHM/2, color='g', linestyle='--')
    ax.axvline(x=mu_fit + FWHM/2, color='g', linestyle='--')
    plt.grid()
    plt.legend(loc="upper left")
    plt.savefig("/mnt/c/Users/mas19/OneDrive/Uni/1.Semester/FP-Master/Protokolle/FP/V44/vXXX/plots/Detectorscan.pdf")
    plt.clf()
#Detectorf(Detector)
def Xscanf(xscan):
    coef1=np.polyfit(xscan[6:9,0],xscan[6:9,1],1)
    slope1, intercept1 = coef1
    x1=np.linspace(xscan[6,0]-1,xscan[8,0]+1,1000)
    y1=slope1*x1+intercept1
    coef2=np.polyfit(xscan[-12:-9,0],xscan[-12:-9,1],1)
    slope2, intercept2 = coef2
    x2=np.linspace(xscan[-12,0]-1,xscan[-9,0]+1,1000)
    y2=slope2*x2+intercept2
    HM=(max(xscan[:,1])+min(xscan[:,1]))/2
    XL=(HM-intercept1)/slope1
    XR=(HM-intercept2)/slope2
    print("Probenbreite=",XR-XL)
    x3=np.linspace(XL,XR,1000)
    fig, ax = plot.Plot6(xscan[:,0], xscan[:,1], r"$x$/mm", r"Intensität $I[1/s]$","X-Scan")
    ax.plot(x1,y1, color='k', linestyle='--', label="Linker Probenrand")
    ax.plot(x2,y2, color='k', linestyle='--', label="Rechter Probenrand")
    ax.plot(x3,HM*np.ones(1000), color='y', linestyle='--', label="Probenbreite= {:.2f} mm".format(XR-XL))
    plt.ylim(1.5e5,4e5)
    plt.grid()
    plt.legend(loc="upper center")
    plt.savefig("/mnt/c/Users/mas19/OneDrive/Uni/1.Semester/FP-Master/Protokolle/FP/V44/vXXX/plots/Xscan.pdf")
    plt.clf()
Xscanf(xscan)