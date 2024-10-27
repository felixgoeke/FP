import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.signal import find_peaks
import os

SKIP_ANFANG = 12
SKIP_ENDE = 14

#Europiumdaten einlesen
europium = pd.read_csv("/home/felix/Arbeitsheft/FP/V18/data/Europium.Spe", skiprows=SKIP_ANFANG, header=None)
europium = europium.iloc[:-SKIP_ENDE]
europium.columns = ["Daten"]

print(europium.columns)  # To see the current column names

#untergrund
untergrund = pd.read_csv("/home/felix/Arbeitsheft/FP/V18/data/Untergrund.Spe", skiprows=SKIP_ANFANG, header=None)
untergrund = untergrund.iloc[:-SKIP_ENDE]
untergrund.columns = ['daten']

europium["data"] = pd.to_numeric(europium["Daten"], errors="coerce")
untergrund["daten"] = pd.to_numeric(untergrund["daten"], errors="coerce")

print(europium.columns.tolist())

#index
europium["index"] = europium.index
untergrund["index"] = untergrund.index

#literaturwerte
europium_lit = pd.read_csv("/home/felix/Arbeitsheft/FP/V18/data/Eu_Lit.csv", skiprows=1,sep='\s+')
europium_lit.columns = ["Energie", "Unischerheit(Energie)", "Intensität", "Unischerheit(Intensität)"]

print(europium_lit.head())  # To view the first few rows and columns of europium_lit
print(europium_lit.columns)  # To see the current column names

#normierung des untergrundes
#untergrundmessung dauerte 78545s, europiummessung 3718s
untergrund["daten"] = untergrund["daten"] / 78545 * 3718

#untergrund abziehen
europium["data"] = europium["data"] - untergrund["daten"]

#negativwerte entfernen
europium = europium["data"].clip(lower=0)

#peaks bestimmen
peaks_array, peaks_params = find_peaks(europium["data"], height=15, prominence=20 ,distance=15)
peaks = pd.DataFrame(peaks_params)
peaks["peaks"] = peaks_array

print(peaks.head())

#untergrund peaks entfernen
#peaks = peaks.drop([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20])

#plot
plt.figure(figsize=(10,5))

plt.bar(europium["index"], europium["data"], linewidth=2, label= r'$^{152}\mathrm{EU}$',color="blue", alpha=0.5)
plt.plot(peaks["peaks"], peaks["peak_heights"], "x", color="red", label="Peaks")

for peak in peaks["peaks"]:
    plt.axvline(x=peak, color='orange', linestyle='--', alpha=0.7)


plt.xlabel("Kanal")
plt.ylabel("Counts")

plt.legend()
plt.grid(True)

os.makedirs('../plots', exist_ok=True)
plt.savefig("../plots/europium.pdf")