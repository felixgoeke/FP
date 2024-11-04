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
    caesium["daten"], height=15, prominence=30, distance=10
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
# Den ersten Peak weglassen
for peak in peaks["peaks"].iloc[1:]:
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
print(f"Photo-Peak bei Kanal {peaks['peaks'][1]}")
photo_energie = peaks["peaks"][1] * alpha + beta
print(f"Photo-Peak-Energie: {photo_energie} keV (irgendwie um faktor 2 falsch)")

# Integration der Fit-Funktion des Photopeaks
def integrate_gauss(A, mu, sigma):
    integral, _ = quad(lambda x: gauss(x, A, mu, sigma), x_data.min(), x_data.max())
    return integral

# Berechnung des Inhalts des Photopeaks
photo_peak_A = m.values["A"]
photo_peak_mu = m.values["mu"]
photo_peak_sigma = m.values["sigma"]

photo_peak_content = integrate_gauss(photo_peak_A, photo_peak_mu, photo_peak_sigma)
print(f"Inhalt des Photopeaks: {photo_peak_content}")
#Inhalt des Photopeaks: 11525.206774769158

#Compton-Kante
compton_kante_theorie = compton_kante(662)
print(f"Theoretische Compton-Kante bei {compton_kante_theorie} keV")
compton_kante_energie = compton_kante(photo_energie)
print(f"Compton-Kante bei {compton_kante_energie} keV")
#Compton-Kante mit der halben Energie des Photo-Peaks
compton_kante_energie2 = compton_kante(photo_energie / 2)
print(f"Compton-Kante 1/2 bei {compton_kante_energie2} keV") #hier kommt das richtige raus

#Umrechnung wieder zurück in Kanal
def linear_invers(E, alpha, beta):
    #Zur Umrechnung von Energie in Kanälen
    K = E / alpha - beta
    return K
compton_kante_kanal = linear_invers(compton_kante_energie, alpha.n, beta.n)
print(f"Compton-Kante bei Kanal {compton_kante_kanal}")
compton_kante_kanal2 = linear_invers(compton_kante_energie2, alpha.n, beta.n)
print(f"Compton-Kante 1/2 bei Kanal {compton_kante_kanal2}")
compton_kante_kanal_theorie = linear_invers(compton_kante_theorie, alpha.n, beta.n)
print(f"Theoretische Compton-Kante bei Kanal {compton_kante_kanal_theorie}")

#Plot der Compton-Kante und des Backscattering-Peaks
plt.figure(figsize=(10, 5))
# Plot des Caesium-Spektrums
plt.bar(caesium["index"], caesium["daten"], linewidth=2, width=1.1, label=r"$^{137}\mathrm{Cs}$", color="royalblue")

# Plot der Compton-Kante 
plt.axvline(x=compton_kante_kanal.n, color='green', linestyle='--', label=f'Compton-Kante (Berechnet)')
#plt.axvline(x=compton_kante_kanal_theorie, color='purple', linestyle='--', label=f'Compton-Kante (Theorie)')
## Plot der Compton-Kante 1/2
#plt.axvline(x=compton_kante_kanal2.n, color='green', linestyle='--', label=f'Compton-Kante Hälfte: {compton_kante_kanal2.n:.2f} keV')


# Plot des Backscatter-Peaks
backscatter_peak = peaks["peaks"][0]
plt.axvline(x=backscatter_peak, color='red', linestyle='--', label=f'Backscatter-Peak: {backscatter_peak}')

# Bereich zwischen Backscatter-Peak und Compton-Kante 2 farbig unterlegen
plt.fill_betweenx(
    y=[0, caesium["daten"].max()],
    x1=backscatter_peak,
    x2=compton_kante_kanal.n,
    color='gray',
    alpha=0.3,
    label='Backscatter to Compton-Kante 1/2'
)

# Plot-Einstellungen
plt.xlabel(r"Channels")
plt.ylabel(r"Signals")
plt.legend()
plt.grid(True, linewidth=0.1)
plt.tight_layout()
plt.savefig("./plots/Caesium-Compton-Backscatter.pdf")
plt.clf()

#Compton-Kontinuum Fit
def compton_kontinuum(x, a, b):
    return a * x + b

#Bereich um die Compton-Kante herum
x_min = backscatter_peak+300
x_max = round(compton_kante_kanal.n)-500
x_data = caesium["index"][x_min:x_max]
y_data = caesium["daten"][x_min:x_max]

#LeastSquares-Kostenfunktion definieren
least_squares = LeastSquares(x_data, y_data, np.sqrt(y_data), compton_kontinuum)

#Minuit-Objekt erstellen und anpassen
m = Minuit(least_squares, a=0, b=0)
m.migrad()

#Fit-Ergebnisse extrahieren
a_fit, b_fit = ufloat(m.values["a"],m.errors["a"]), ufloat(m.values["b"],m.errors["b"])
print(m)
print(f"Compton-Kontinuum: a = {a_fit}, b = {b_fit}")

#Plot des Compton-Kontinuums
plt.figure(figsize=(10, 5))
#plt.plot(x_data, y_data, "x", label="Data", color="royalblue")
plt.bar(x_data,y_data,linewidth=2, width=1.1, label="Data", color="royalblue")
plt.plot(x_data, compton_kontinuum(x_data, a_fit.n, b_fit.n), color="orange",linewidth=2 ,label="Compton-Kontinuum Fit")
plt.xlabel("Channels")
plt.ylabel("Signals")
plt.legend()
plt.grid(True, linewidth=0.1)
plt.tight_layout()
plt.savefig("./plots/Caesium-Compton-Kontinuum.pdf")
plt.clf()

# Bestimme den Inhalt des Compton-Kontinuums
def integrate_compton_kontinuum(a, b, backscatter_peak, x_max):
    integral, _ = quad(lambda x: compton_kontinuum(x, a, b), backscatter_peak, x_max)
    return integral

# Berechnung des Inhalts des Compton-Kontinuums
compton_kontinuum_content = integrate_compton_kontinuum(a_fit.n, b_fit.n, backscatter_peak, round(compton_kante_kanal.n))
print(f"Inhalt des Compton-Kontinuums: {compton_kontinuum_content}")


#Compton-Kontinuum: a = 0.00183+/-0.00009, b = 2.13+/-0.30
#Inhalt des Compton-Kontinuums: 26105.178706110488

#Verhältnis Compton-Kontinuum zu Photo-Peak
compton_photo_verhaeltnis = compton_kontinuum_content / photo_peak_content
print(f"Das Verhältnis zwischen dem Comptoninhalt und dem Photopeakinhalt ist: {compton_photo_verhaeltnis:.3f}")

#Theorertischer Wert des Rückstreu-Peaks
def backscatter(E_gamma):
    return E_gamma / ( 1 + 2 * (E_gamma / 510.9989))

backscatter_peak_theorie = backscatter(photo_energie)
backscatter_peak_theorie_halbe = backscatter(photo_energie/2)

print(f"Der theoretischer Backscatter-Peak liegt bei {backscatter_peak_theorie:.3}")
print(f"Der theoretischer Backscatter-Peak (halbe) liegt bei {backscatter_peak_theorie_halbe:.3}")

def absorption(mu,d):
    return (1 - np.exp(-mu*d))*100 #in Prozent

mu_photo =0.008
mu_compton = 0.37

d=3.9

absorption_photo = absorption(mu_photo,d)
absorption_compton = absorption(mu_compton,d)

print(f"Die Absorptionswahscheinlichkeit für den Photopeak ist:{absorption_photo:.3}")
print(f"Die Absorptionswahscheinlichkeit für das Comptonkontinuum ist:{absorption_compton:.3}")