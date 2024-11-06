import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import fsolve
from scipy.stats import norm
from scipy.integrate import quad
import scipy
import iminuit
from iminuit import Minuit
from iminuit.cost import LeastSquares
from typing import Tuple
from uncertainties import ufloat
import os

# Einlesen der Uranmessung
SKIP_ANFANG = 12
SKIP_ENDE = 14

uran = pd.read_csv("./data/Unbekannt(Uran).Spe", skiprows=SKIP_ANFANG, header=None)
uran = uran.iloc[:-SKIP_ENDE]  # Entferne den Rotz am Ende
uran.columns = ["Daten"]

# Einlesen der Untergrundmessung
untergrund = pd.read_csv("./data/Untergrund.Spe", skiprows=SKIP_ANFANG, header=None)
untergrund.columns = ["daten"]
untergrund = untergrund.iloc[:-SKIP_ENDE]

# Sicherstellen, dass die Spalte "data" numerische Werte enthält
uran["daten"] = pd.to_numeric(uran["Daten"], errors="coerce")
untergrund["daten"] = pd.to_numeric(untergrund["daten"], errors="coerce")

# Index ergänzen
uran["index"] = uran.index
untergrund["index"] = untergrund.index

#normierung des untergrundes
#untergrundmessung dauerte 78545s, uranmessung 3427s
#untergrund["daten"] = untergrund["daten"] * (3427 / 78545)
#untergrund von uran abziehen
uran["daten"] = uran["daten"] - (untergrund["daten"]* (3427 / 78545))

# Negative Werte in einem Histogramm sind unphysikalisch
uran["daten"] = uran["daten"].clip(lower=0)

# Finden der Peaks
peaks_array, peaks_params = find_peaks(
    uran["daten"], height=20, prominence=50, distance=100
)
peaks = pd.DataFrame(peaks_params)
peaks["peaks"] = peaks_array

#Müll entfernen
peaks = peaks.drop([0,1,2,7,11,16,17,21,22])

peaks.to_csv("./data/uran-peaks.csv", index=False)
print("Saving uran-peaks.csv")

# Plot der Barium-Daten
plt.figure(figsize=(21, 9))
plt.bar(uran["index"], uran["daten"], linewidth=2, width=1.1, label=r"$^{133}\mathrm{Ba}$", color="royalblue")
plt.bar(untergrund["index"], untergrund["daten"], linewidth=2, width=1.1, label="Untergrund", color="orange")
plt.plot(peaks["peaks"], peaks["peak_heights"], "x", color="red", label="Peaks")
plt.xlabel(r"Channels")
plt.ylabel(r"Signals")
plt.title(r"Uran Data")
plt.legend()
plt.grid(True, linewidth=0.1)
plt.savefig("./plots/Uran.pdf")
plt.clf()
print("Plot wurde gespeichert")

#Umrechnen der Kanäle in Energie
def kanal_energie(kanal, alpha, beta):
    """Formel zur Umrechnung von Kanälen in Energie"""
    E = alpha * kanal + beta
    return E

#Daten aus der Kalibrierung
alpha = ufloat(0.228334, 0)
beta  = ufloat(-148.659709, 0.000593)

#Energie der Peaks
peaks["Energie 1/2"] = kanal_energie(peaks["peaks"], alpha, beta) /2
peaks["Energie"] = kanal_energie(peaks["peaks"], alpha, beta)
peaks.to_csv("./data/uran-peaks.csv", index=False)
print("Saving uran-peaks.csv")
