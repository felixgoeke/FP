import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from iminuit import Minuit
from iminuit.cost import LeastSquares
import os
from datetime import datetime
from uncertainties import ufloat
from uncertainties.unumpy import exp as uexp

aktivität_0 = ufloat(4130,60) # Bq am 1.10.2000
start_datum = datetime(2000, 10, 1)
end_datum = datetime(2024, 10, 21)
time_diff = (end_datum - start_datum).total_seconds() #Zeitdiffernz zwischen Produktion und Experiment
print("Die Zeitdifferenz beträgt:", time_diff)

#halbwertszeit_eu = 426.7 * 10**6 # Halbwertszeit in Sekunden
halbwertszeit_eu = 13.522 * 365.25 * 24 * 60 * 60 # Halbwertszeit in Sekunden
print("Die Halbwertszeit beträgt:", halbwertszeit_eu)
decay_constant = np.log(2) / halbwertszeit_eu
t = 3634 #messzeit in sekunden
a,r = 7.02 + 1.5, 4.5/2 #in cm #abmessung detektor: platzhalter 7,02cm, +1,5cm Aluminium
omega = 1/2*(1-a/np.sqrt(a**2+r**2))
print(omega)


#aktivität
def aktivitätsgesetz(t,aktivität_0,decay_constant):
    A = aktivität_0 * uexp(-decay_constant*t)
    return A

end_aktivität = aktivitätsgesetz(time_diff,aktivität_0,decay_constant)
print("Die Endaktivität beträgt:", end_aktivität)

#Vollenergiewahrscheinlichkeitsfunktion
def vollenergiewkeit(omega,N,A,W,t):
    if omega == 0 or t == 0 or A == 0 or W == 0:
        return ValueError("Division durch Null")
    Q = (4*np.pi) / omega*N / (A*W*t)
    return Q

#Peak Daten laden und Detektionswahrscheinlichkeit berechnen
peaks = pd.read_csv("./build/peaks.csv")
peaks["fedp"], peaks["N"], peaks["N_err"] = float(0), float(0), float(0) #Vollernergienachweiswahrscheinlichkeit


for index, row in peaks.iterrows():
    N = ufloat(row["N"],row["N_err"])
    P = ufloat(row["Intensität"],row["Unsicherheit(I)"])
    try:
        # Calculate detection probability
        u_fedp = vollenergiewkeit(omega*4*np.pi, N, end_aktivität, P, t)
        peaks.at[index, "fedp"] = round(u_fedp.n,6) * 10  # Use nominal value of u_fedp
        peaks.at[index, "fedp_err"] = u_fedp.std_dev * 10    # Use std dev of u_fedp
    except ValueError as e:
        print(f"Error at index {index}: {e}")
        peaks.at[index, "fedp"], peaks.at[index, "fedp_err"] = np.nan, np.nan  # Assign NaN for error cases

peaks.to_csv("./build/peaks.csv",index=False)

def energie(E,a,b):
    return a*(E**b)

least_squares = LeastSquares(peaks["Energie"],peaks["fedp"],peaks["fedp_err"],energie)
m = Minuit(least_squares, a=4, b=-0.9)
m.limits["b"] = (-1,-0.8)
m.migrad()
m.hesse()

#Plotten
energy_scale = np.linspace(peaks["Energie"].min(),peaks["Energie"].max(),1000)
fit_curve = energie(energy_scale, *m.values)

plt.figure(figsize=(10,6))
plt.errorbar(peaks["Energie"],peaks["fedp"],peaks["fedp_err"],fmt="o",label="Messwerte")
plt.plot(energy_scale,fit_curve,label="Fit",color="orange")
plt.xlabel("Energie [keV]")
plt.ylabel("Nachweiswahrscheinlichkeit")
plt.title("Nachweiswahrscheinlichkeit in Abhängigkeit der Energie")
plt.legend()
plt.grid()
plt.savefig("./plots/Nachweiswahrscheinlichkeit.pdf")
