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

# Einlesen der Bariummessung
SKIP_ANFANG = 12
SKIP_ENDE = 14

barium = pd.read_csv("./data/Barium.Spe", skiprows=SKIP_ANFANG, header=None)
barium = barium.iloc[:-SKIP_ENDE]  # Entferne den Rotz am Ende
barium.columns = ["Daten"]

# Einlesen der Untergrundmessung
untergrund = pd.read_csv("./data/Untergrund.Spe", skiprows=SKIP_ANFANG, header=None)
untergrund.columns = ["daten"]
untergrund = untergrund.iloc[:-SKIP_ENDE]

# Sicherstellen, dass die Spalte "data" numerische Werte enthält
barium["daten"] = pd.to_numeric(barium["Daten"], errors="coerce")
untergrund["daten"] = pd.to_numeric(untergrund["daten"], errors="coerce")

# Index ergänzen
barium["index"] = barium.index
untergrund["index"] = untergrund.index

#normierung des untergrundes
#untergrundmessung dauerte 78545s, bariummessung 3021s
untergrund["daten"] = untergrund["daten"] * (3021 / 78545)
#untergrund von barium abziehen
barium["daten"] = barium["daten"] - untergrund["daten"]

# Negative Werte in einem Histogramm sind unphysikalisch
barium["daten"] = barium["daten"].clip(lower=0)

# Finden der Peaks
peaks_array, peaks_params = find_peaks(
    barium["daten"], height=15, prominence=30, distance=10
)
peaks = pd.DataFrame(peaks_params)
peaks["peaks"] = peaks_array

# Peaks in der Nähe von 0 entfernen
peaks = peaks.drop([0])

# Plot der Barium-Daten
plt.figure(figsize=(21, 9))
plt.bar(barium["index"], barium["daten"], linewidth=2, width=1.1, label=r"$^{133}\mathrm{Ba}$", color="royalblue")
plt.plot(peaks["peaks"], peaks["peak_heights"], "x", color="red", label="Peaks")
plt.xlabel(r"Channels")
plt.ylabel(r"Signals")
plt.title(r"Barium Data")
plt.grid(True, linewidth=0.1)
plt.savefig("./plots/Barium.pdf")
plt.clf()
print("Plot wurde gespeichert")


# Gauss-Funktion
# Fitfunktion für die Peaks
def gauss(x, A, mu, sigma):
    return (A / (np.sqrt(2 * np.pi) *sigma)) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

#Inhalt des Peaks
def integrate_gauss(A, mu, sigma):
    integral, _ = quad(lambda x: gauss(x, A, mu, sigma), x_data.min(), x_data.max())
    return integral

# Fit der Peaks
# Fitten der Peaks mit der Gauss-Funktion
for peak in peaks["peaks"][:-1]:  # Letzten Peak weglassen
    # Bereich um den Peak herum definieren
    window = 20
    x_data = barium["index"][peak - window:peak + window]
    y_data = barium["daten"][peak - window:peak + window]

    # LeastSquares-Kostenfunktion definieren
    least_squares = LeastSquares(x_data, y_data, np.sqrt(y_data), gauss)

    # Minuit-Objekt erstellen und anpassen
    m = Minuit(least_squares, A=y_data.max(), mu=peak, sigma=5)
    m.migrad()

    # Fit-Ergebnisse extrahieren
    A_fit, mu_fit, sigma_fit = ufloat(m.values["A"],m.errors["A"]), ufloat(m.values["mu"],m.errors["mu"]), ufloat(m.values["sigma"],m.errors["sigma"])
    #print(m)
    print(f"Peak at {peak}: A = {A_fit}, mu = {mu_fit}, sigma = {sigma_fit}")

    # Berechnung des Inhalts des Photopeaks
    photo_peak_A = m.values["A"]
    photo_peak_mu = m.values["mu"]
    photo_peak_sigma = m.values["sigma"]

    photo_peak_content = integrate_gauss(photo_peak_A, photo_peak_mu, photo_peak_sigma)
    print(f"Inhalt des {peak}-Photopeaks: {photo_peak_content}")

    # Speichern der Fit-Daten in einer Datei
    with open("./data/barium_fit_data.txt", "a") as f:
        f.write(f"Peak at {peak}: A = {A_fit}, mu = {mu_fit}, sigma = {sigma_fit}, Inhalt:{photo_peak_content}\n")
 
    # Plotten des Fits und der Daten
    plt.figure(figsize=(10, 5))
    #plt.bar(x_data, y_data, linewidth=2, width=1.1,alpha=0.2, label="Data", color="royalblue")
    plt.plot(x_data, y_data, "x", label="Data", color="royalblue")
    plt.plot(x_data, gauss(x_data, A_fit.n, mu_fit.n, sigma_fit.n), color="orange", label="Gaussian Fit")
    plt.xlabel("Channels")
    plt.ylabel("Signals")
    plt.legend()
    #plt.title(f"Peak at {peak}")
    plt.grid(True, linewidth=0.1)
    plt.tight_layout()
    plt.savefig(f"./plots/Barium-Peak-{peak}.pdf")
    plt.clf()

# Focused data extraction for a better fit
peak = peaks["peaks"].iloc[-1]
window = 15  # Smaller window to focus on the peak area
x_data = barium["index"][peak - window:peak + window]
y_data = barium["daten"][peak - window:peak + window]

# Improved initial guesses
A_initial = y_data.max()  # Amplitude set to max of y_data
mu_initial = np.average(x_data, weights=y_data)  # Weighted mean of x_data
sigma_initial = (x_data.max() - x_data.min()) / 6  # Initial guess for sigma

# LeastSquares cost function
least_squares = LeastSquares(x_data, y_data, np.sqrt(y_data), gauss)

# Minuit object with improved initial parameters
m = Minuit(least_squares, A=A_initial, mu=mu_initial, sigma=sigma_initial)
m.migrad()  # Adding tolerance for higher precision

# Extract fitted parameters with uncertainties
A_fit, mu_fit, sigma_fit = ufloat(m.values["A"], m.errors["A"]), ufloat(m.values["mu"], m.errors["mu"]), ufloat(m.values["sigma"], m.errors["sigma"])
#print(m)
print(f"Peak at {peak}: A = {A_fit}, mu = {mu_fit}, sigma = {sigma_fit}")

# Berechnung des Inhalts des Photopeaks
photo_peak_A = m.values["A"]
photo_peak_mu = m.values["mu"]
photo_peak_sigma = m.values["sigma"]

photo_peak_content = integrate_gauss(photo_peak_A, photo_peak_mu, photo_peak_sigma)
print(f"Inhalt des {peak}-Photopeaks: {photo_peak_content}")

# Speichern der Fit-Daten in einer Datei
with open("./data/barium_fit_data.txt", "a") as f:
    f.write(f"Peak at {peak}: A = {A_fit}, mu = {mu_fit}, sigma = {sigma_fit}, Inhalt:{photo_peak_content}\n ")

# Plot the data and fit
plt.figure(figsize=(10, 5))
plt.plot(x_data, y_data, "x", label="Data", color="royalblue")
plt.plot(x_data, gauss(x_data, A_fit.n, mu_fit.n, sigma_fit.n), color="orange", label="Gaussian Fit")
plt.xlabel("Channels")
plt.ylabel("Signals")
plt.legend()
plt.title(f"Peak at {peak} Version 2")
plt.grid(True, linewidth=0.1)
plt.tight_layout()
plt.savefig(f"./plots/Barium-Peak-{peak}-V2.pdf")
plt.clf()

#Peak at 793: A = (5.61+/-0.08)e+03, mu = 791.54+/-0.10, sigma = 6.99+/-0.09, Inhalt:5580.536937037902
#Peak at 2685: A = 846+/-29, mu = 2685.45+/-0.26, sigma = 7.37+/-0.20, Inhalt:839.0030383156169
#Peak at 2940: A = (1.88+/-0.04)e+03, mu = 2941.44+/-0.16, sigma = 6.78+/-0.11, Inhalt:1868.217984644541
#Peak at 3458: A = (5.20+/-0.07)e+03, mu = 3456.54+/-0.11, sigma = 7.53+/-0.08, Inhalt:5142.752649460566
#Peak at 3730: A = 641+/-27, mu = 3726.82+/-0.34, sigma = 7.02+/-0.30, Inhalt:607.0061981689814
 
inhalt_793 = 5580.536937037902
inhalt_2685 = 839.0030383156169
inhalt_2940 = 1868.217984644541
inhalt_3458 = 5142.752649460566
inhalt_3730 = 607.0061981689814

#Energiebestimmung aus Kanalnummer
def linear(K, alpha, beta):
    return alpha * K + beta

# Das hier sind die Fitparameter aus der Kalibrierung
alpha = ufloat(0.228334, 0)
beta  = ufloat(-148.659709, 0.000593)

# Energie der Peaks bestimmen
energie_793 = linear(793, alpha, beta)
energie_2685 = linear(2685, alpha, beta)
energie_2940 = linear(2940, alpha, beta)
energie_3458 = linear(3458, alpha, beta)
energie_3730 = linear(3730, alpha, beta)

#Effizienz bestimmen
def q_energy(e, a, b):
    """Formel für die Energieabhängigkeit der Vollenergienachweiswahrscheinlichkeit"""
    Q = a * e** b 
    return Q

# Fitparameter für die Effizienz
#Fit-Ergebnisse: a = 19.84894362204279 ± 2.887590018432524, b = -1.3343701493052662 ± 0.026767376407957545
a = ufloat(19.84894362204279, 2.887590018432524)
b = ufloat(-1.3343701493052662, 0.026767376407957545)

#Effizienz der Peaks bestimmen
effizienz_793 = q_energy(energie_793, a, b)
effizienz_2685 = q_energy(energie_2685, a, b)
effizienz_2940 = q_energy(energie_2940, a, b)
effizienz_3458 = q_energy(energie_3458, a, b)
effizienz_3730 = q_energy(energie_3730, a, b)

print(f"Effizienz des {energie_793}-keV-Peaks: {effizienz_793}")
print(f"Effizienz des {energie_2685}-keV-Peaks: {effizienz_2685}")
print(f"Effizienz des {energie_2940}-keV-Peaks: {effizienz_2940}")
print(f"Effizienz des {energie_3458}-keV-Peaks: {effizienz_3458}")
print(f"Effizienz des {energie_3730}-keV-Peaks: {effizienz_3730}")

#Aktivität ausrechnen
t = 3021 #in s
omega_4pi = 0.01657
p_793 = ufloat(33.31, 0.30) #in prozent
p_2685 = ufloat(7.13,0.06) #in prozent
p_2940 = ufloat(18.31, 0.11) #in prozent
p_3458 = ufloat(62.05, 0.19) #in prozent
p_3730 = ufloat(8.94, 0.06) #in prozent

def aktivitaet(inhalt, effizienz, p, omega,t):
    return inhalt * omega / (effizienz * (0.01)* p * t)

aktivitaet_793 = aktivitaet(inhalt_793, effizienz_793, p_793, omega_4pi,t)*10
aktivitaet_2685 = aktivitaet(inhalt_2685, effizienz_2685, p_2685, omega_4pi,t)*10
aktivitaet_2940 = aktivitaet(inhalt_2940, effizienz_2940, p_2940, omega_4pi,t)*10
aktivitaet_3458 = aktivitaet(inhalt_3458, effizienz_3458, p_3458, omega_4pi,t)*10
aktivitaet_3730 = aktivitaet(inhalt_3730, effizienz_3730, p_3730, omega_4pi,t)*10

print(f"Aktivität des {energie_793}-keV-Peaks: {aktivitaet_793}")
print(f"Aktivität des {energie_2685}-keV-Peaks: {aktivitaet_2685}")
print(f"Aktivität des {energie_2940}-keV-Peaks: {aktivitaet_2940}")
print(f"Aktivität des {energie_3458}-keV-Peaks: {aktivitaet_3458}")
print(f"Aktivität des {energie_3730}-keV-Peaks: {aktivitaet_3730}")

