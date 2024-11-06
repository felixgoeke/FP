import pandas as pd
from scipy.integrate import quad
from uncertainties import ufloat
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import numpy as np
from scipy.signal import find_peaks
import iminuit
from iminuit import Minuit
from iminuit.cost import LeastSquares
import os

matplotlib.rcParams.update({"font.size": 18})

# Einlesen der Kalibrationsmessung
SKIP_ANFANG = 12
SKIP_ENDE = 14

europium = pd.read_csv("./data/Europium.Spe", skiprows=SKIP_ANFANG, header=None)
europium = europium.iloc[:-SKIP_ENDE]  # Entferne den Rotz am Ende
europium.columns = ["Daten"]

# Einlesen der Untergrundmessung
untergrund = pd.read_csv("./data/Untergrund.Spe", skiprows=SKIP_ANFANG, header=None)
untergrund.columns = ["daten"]
untergrund = untergrund.iloc[:-SKIP_ENDE]

# Sicherstellen, dass die Spalte "data" numerische Werte enthält
europium["data"] = pd.to_numeric(europium["Daten"], errors="coerce")
untergrund["daten"] = pd.to_numeric(untergrund["daten"], errors="coerce")

# Index ergänzen
europium["index"] = europium.index
untergrund["index"] = untergrund.index

# Einlesen der Literaturwerte
europium_lit = pd.read_csv(
    "./data/Europium_Lit.csv",
    sep=";",
    skiprows=13,
    header=None,
    usecols=[0, 1, 2, 3]
)
europium_lit.columns = ["Energie", "Unsicherheit(E)", "Intensität", "Unsicherheit(I)"]
europium_lit["Energie"] = pd.to_numeric(europium_lit["Energie"], errors="coerce")
europium_lit["Unsicherheit(E)"] = pd.to_numeric(europium_lit["Unsicherheit(E)"], errors="coerce")
europium_lit["Intensität"] = pd.to_numeric(europium_lit["Intensität"], errors="coerce")
europium_lit["Unsicherheit(I)"] = pd.to_numeric(europium_lit["Unsicherheit(I)"], errors="coerce")

#normierung des untergrundes
#untergrundmessung dauerte 78545s, europiummessung 3718s
untergrund["daten"] = untergrund["daten"] * (3718 / 78545)

# Plot der Europium-Daten
plt.figure(figsize=(21, 9))
plt.bar(europium["index"], europium["data"], linewidth=2, width=1.1, label=r"$^{152}\mathrm{Eu}$", color="royalblue")
plt.bar(untergrund["index"], untergrund["daten"], linewidth=2, width=1.1, label="Untergrund", color="orange")
plt.xlabel(r"Channels")
plt.ylabel(r"Signals")
plt.title(r"Europium Data")
plt.grid(True, linewidth=0.1)
plt.legend()
plt.tight_layout()
plt.savefig("./plots/Europium.pdf")
plt.clf()

# Untergrund entfernen
europium["data"] = europium["data"] - untergrund["daten"]


# Negative Werte in einem Histogramm sind unphysikalisch
europium["data"] = europium["data"].clip(lower=0)

# Daten im Bereich von Zeile 1000 bis 8000 betrachten, ohne sie tatsächlich abzuschneiden
europium_view = europium.iloc[79:8000]
untergrund_view = untergrund.iloc[79:8000]

# Peaks bestimmen und mit den zugehörigen Parametern in Dataframe speichern
peaks_array, peaks_params = find_peaks(
    europium_view["data"], height=5, prominence=15, distance=50
)
peaks = pd.DataFrame(peaks_params)
peaks["peaks"] = peaks_array +79 # Offset durch 900 Zeilen

#droppe peaks die dem Untergrund zuzuordnen sind
peaks = peaks.drop([0,1, 2, 3, 4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,20,21,22,23,24,25,26,27])

europium_lit = europium_lit.head(len(peaks))
# Plot der Kalibrationsmessung
plt.figure(figsize=(21, 9))

plt.bar(
    europium["index"],
    europium["data"],
    linewidth=2,
    width=1.1,
    label=r"$^{152}\mathrm{Eu}$",
    color="royalblue",
)
plt.plot(peaks["peaks"], peaks["peak_heights"], "x", color="orange", label="Peaks")

plt.xticks(np.linspace(0, 8191, 10))
plt.yticks(np.linspace(europium["data"].min(), europium["data"].max(), 10))

plt.ylim(europium["data"].min() - 30)

plt.xlabel(r"Kanal")
plt.ylabel(r"Signal")

plt.grid(True, linewidth=0.1)
plt.legend()
plt.tight_layout()
plt.savefig("./plots/Europium-Peaks.pdf")
plt.clf()

#erstelle eine Spalte "Inhalt" im Peaks-DataFrame, falls sie noch nicht existiert
if "Inhalt" not in peaks.columns:
    peaks["Inhalt"] = np.nan
if "Inhalt Error" not in peaks.columns:
    peaks["Inhalt Error"] = np.nan

# Fitfunktion für die Peaks
def gauss(x, A, mu, sigma):
    return (A / (np.sqrt(2 * np.pi) *sigma)) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

#Inhalt des Peaks
def integrate_gauss(A, mu, sigma):
    integral, _ = quad(lambda x: gauss(x, A, mu, sigma), x_data.min(), x_data.max())
    return integral

# Fitten der Peaks mit der Gauss-Funktion
# Den ersten Peak weglassen
for peak in peaks["peaks"]:
    # Bereich um den Peak herum definieren
    window = 20
    x_data = europium["index"][peak - window:peak + window]
    y_data = europium["data"][peak - window:peak + window]

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

    photo_peak_content = integrate_gauss(A_fit.n, mu_fit.n, sigma_fit.n)
    print(f"Inhalt des {peak}-Photopeaks: {photo_peak_content}")
    # Speichern der Peak-Daten in einer CSV-Datei
    # Erstelle eine Spalte "Inhalt" im Peaks-DataFrame, falls sie noch nicht existiert

    # Speichere den Photopeak-Inhalt in der zum Peak passenden Zeile
    peaks.loc[peaks["peaks"] == peak, "Inhalt"] = round(photo_peak_content, 1)
    peaks.loc[peaks["peaks"] == peak, "Inhalt Error"] = round(np.sqrt(photo_peak_content),1)

    # Speichern der Peak-Daten in einer CSV-Datei
    #peaks.to_csv("./build/peaks.csv", index=False)

    # Plotten des Fits und der Daten
    plt.figure(figsize=(10, 5))
    #plt.bar(x_data, y_data, linewidth=2, width=1.1,alpha=0.2, label="Data", color="royalblue")
    plt.plot(x_data, y_data, "x", label="Data", color="royalblue")
    plt.plot(x_data, gauss(x_data, A_fit.n, mu_fit.n, sigma_fit.n), color="orange", label="Gaussian Fit")
    plt.xlabel("Kanal")
    plt.ylabel("Signal")
    plt.legend()
    #plt.title(f"Peak at {peak}")
    plt.grid(True, linewidth=0.1)
    plt.tight_layout()
    plt.savefig(f"./plots/Europium-Peak-{peak}.pdf")
    plt.clf()

#händische Countzählung für Peak 4303
peaks.loc[peaks["peaks"] == 4303, "Inhalt"] =  388.6 
peaks.loc[peaks["peaks"] == 4303, "Inhalt Error"] =  34

# Nach der Kanalnummer aufsteigend sortieren
peaks.sort_values(by="peaks", inplace=True, ascending=True)
europium_lit.sort_values(by="Energie", inplace=True, ascending=True)

# Index beider Dfs zurücksetzen
peaks = peaks.reset_index(drop=True)
europium_lit = europium_lit.reset_index(drop=True)

# Nochmal alles in ein Df damit alles leichter gespeichert werden kann
peaks = pd.concat([europium_lit, peaks], axis=1)


def linear(K, alpha, beta):
    #Fit zwischen Kanalnummer und Energie
    return alpha * K + beta


def linear_invers(E, alpha, beta):
    #Zur Umrechnung von Energie in Kanälen
    K = E / alpha - beta
    return K


least_squares = LeastSquares(
    peaks["peaks"], europium_lit["Energie"], europium_lit["Unsicherheit(E)"], linear
)
m = Minuit(least_squares, alpha=0, beta=0)
m.migrad()
m.hesse()

# print(least_squares.pulls(m.values)) So könnte man die Pulls ploten, sind hier aber gewaltig...
matplotlib.rcParams.update({"font.size": 8})
plt.figure()
plt.plot(
    peaks["peaks"],
    linear(peaks["peaks"], *m.values),
    label="fit",
    color="orange",
    linewidth=2.2,
    zorder=1,
)
plt.errorbar(
    peaks["peaks"],
    europium_lit["Energie"],
    yerr=europium_lit["Unsicherheit(E)"],
    fmt="o",
    label="data",
    color="royalblue",
    elinewidth=2.2,
    barsabove=True,
    zorder=2,
)

fit_info = []  


os.makedirs('build', exist_ok=True)
with open("./build/Fitparameter_Kalib.txt", "w") as file:
    for p, v, e in zip(m.parameters, m.values, m.errors):
        fit_info.append(f"{p} = ${v:.6f} \\pm {e:.6f}$")
        file.write(f"{p} = ${v:.6f} \\pm {e:.6f}$\n")

# Für Weiterbenutzung in anderen Skripten
peaks.to_csv("./build/peaks.csv", index=False)

plt.legend(title="\n".join(fit_info), frameon=False)
plt.xlabel(r"$\mathrm{Kanal}$")
plt.ylabel(r"$\mathrm{Energie}/\mathrm{keV}$")
plt.tight_layout()
plt.savefig("./plots/Europium-Fit.pdf")
plt.clf()