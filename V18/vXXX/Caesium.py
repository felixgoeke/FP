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
#from kalibrierung2 import linear, linear_invers


# Das hier sind die Fitparameter aus der Kalibrierung
alpha = ufloat(0.228334, 0)
beta  = ufloat(-148.659709, 0.000593)

# Einlesen der Caesium Messung
SKIP_ANFANG = 12
SKIP_ENDE = 14

caesium = pd.read_csv("./data/Caesium.Spe", skiprows=SKIP_ANFANG, header=None)
caesium = caesium.iloc[:-SKIP_ENDE]
caesium.columns = ["daten"]

# Einlesen der Untergrundmessung
untergrund = pd.read_csv("./data/Untergrund.Spe", skiprows=SKIP_ANFANG, header=None)
untergrund.columns = ["daten"]
untergrund = untergrund.iloc[:-SKIP_ENDE]

# Sicherstellen, dass die Spalte "data" numerische Werte enthält
caesium["daten"] = pd.to_numeric(caesium["daten"], errors="coerce")
untergrund["daten"] = pd.to_numeric(untergrund["daten"], errors="coerce")

# Index ergänzen
caesium["index"] = caesium.index
untergrund["index"] = untergrund.index

# Untergrundmessung skalieren; 
#untergrundmessung dauerte 78545s, Cs-Messung 3546s
untergrund["daten"] = untergrund["daten"] * (3546 / 78545)

# Untergrund entfernen
caesium["daten"] = caesium["daten"] - untergrund["daten"]

# Negative Werte in einem Histogramm sind unphysikalisch
caesium["daten"] = caesium["daten"].clip(lower=0)

# Peaks raussuchen um sie zu fitten
peaks_array, peaks_params = find_peaks(
    caesium["daten"], height=20, prominence=40, distance=10
)
peaks = pd.DataFrame(peaks_params)
peaks["peaks"] = peaks_array

# Raus mit dem Müll
peaks = peaks.drop([0,1])
peaks = peaks.reset_index(drop=True)

#Plot der Cs-Daten und der Peaks
plt.figure(figsize=(21, 9), dpi=500)
plt.bar(
    caesium["index"],
    caesium["daten"],
    linewidth=2,
    width=1.1,
    label=r"$^{137}\mathrm{Cs}$",
    color="royalblue",
)
plt.plot(peaks["peaks"], peaks["peak_heights"], "x", color="orange", label="Peaks")
plt.xticks(np.linspace(0, 8191, 10))
plt.yticks(np.linspace(caesium["daten"].min(), caesium["daten"].max(), 10))

plt.ylim(caesium["daten"].min() - 30)
plt.xlabel(r"Channels")
plt.ylabel(r"Signals")

plt.grid(True, linewidth=0.1)
plt.legend()
plt.tight_layout()
print("Saving Caesium-Peaks.pdf")
plt.savefig("./plots/Caesium-Peaks.pdf")
plt.clf()

#Berechnung der theoretischen Compton-Kante
def compton_kante(E_gamma):
    return E_gamma / (1 + (510.9989) / (2 * E_gamma))

# Speichern der Peaks-Daten in einer CSV-Datei
peaks.to_csv("./data/caesium-peaks.csv", index=False)
print("Saving caesium-peaks.csv")

# Fitfunktion für die Peaks
def gauss(x, A, mu, sigma):
    return (A / (np.sqrt(2 * np.pi) *sigma)) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

# Fitten der Peaks mit der Gauss-Funktion
for peak in peaks["peaks"]:
    # Bereich um den Peak herum definieren
    window = 30
    x_data = caesium["index"][peak - window:peak + window]
    y_data = caesium["daten"][peak - window:peak + window]

    # LeastSquares-Kostenfunktion definieren
    least_squares = LeastSquares(x_data, y_data, np.sqrt(y_data), gauss)

    # Minuit-Objekt erstellen und anpassen
    m = Minuit(least_squares, A=y_data.max(), mu=peak, sigma=5)
    m.migrad()

    # Fit-Ergebnisse extrahieren
    A_fit, mu_fit, sigma_fit = ufloat(m.values["A"],m.errors["A"]), ufloat(m.values["mu"],m.errors["mu"]), ufloat(m.values["sigma"],m.errors["sigma"])
    print(m)
    print(f"Peak at {peak}: A = {A_fit}, mu = {mu_fit}, sigma = {sigma_fit}")

    # Berechnung der Full Width at Half Maximum (FWHM)
    fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma_fit
    fwhm_x = [mu_fit.n - fwhm.n / 2, mu_fit.n + fwhm.n / 2]
    fwhm_y = [gauss(mu_fit.n, A_fit.n, mu_fit.n, sigma_fit.n) / 2] * 2
    fwhm_energie = fwhm * alpha
    print(f"FWHM: {fwhm}, Energie: {fwhm_energie} keV")
    #ausgabe: 22.77 +- 0.16

    # Berechnung der Zehntelwertsbreite (FWTM)
    fwtm = 2 * np.sqrt(2 * np.log(10)) * sigma_fit
    fwtm_x = [mu_fit.n - fwtm.n / 2, mu_fit.n + fwtm.n / 2]
    fwtm_y = [gauss(mu_fit.n, A_fit.n, mu_fit.n, sigma_fit.n) / 10] * 2
    fwtm_energie = fwtm * alpha
    print(f"FWTM: {fwtm}, Energie: {fwtm_energie} keV")
    #ausgabe: 41.51 +- 0.29

    # Plotten des Fits und der Daten
    plt.figure(figsize=(10, 5))
    #plt.bar(x_data, y_data, linewidth=2, width=1.1,alpha=0.2, label="Data", color="royalblue")
    plt.plot(x_data, y_data, "x", label="Data", color="royalblue")
    plt.plot(x_data, gauss(x_data, A_fit.n, mu_fit.n, sigma_fit.n), color="orange", label="Gaussian Fit")
    plt.plot(fwhm_x, fwhm_y, 'r--', label='FWHM')
    plt.plot(fwtm_x, fwtm_y, 'g--', label='FWTM')
    plt.xlabel("Channels")
    plt.ylabel("Signals")
    plt.legend()
    #plt.title(f"Peak at {peak}")
    plt.grid(True, linewidth=0.1)
    plt.tight_layout()
    plt.savefig(f"./plots/Caesium-Peak-{peak}.pdf")
    plt.clf()

#Verhältnis der FWHM und FWTM
verhaeltnis = fwhm / fwtm
print(f"Verhältnis FWHM/FWTM: {verhaeltnis}")

#Photo-Peak-Energie
print(f"Photo-Peak bei Kanal {peaks['peaks'][0]}")
photo_energie = peaks["peaks"][0] * alpha + beta
print(f"Photo-Peak-Energie: {photo_energie} keV (irgendwie um faktor 2 falsch)")
