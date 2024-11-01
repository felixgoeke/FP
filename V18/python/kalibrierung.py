import numpy as np
from scipy.signal import find_peaks
import pandas as pd
from iminuit.cost import LeastSquares
import iminuit
from iminuit import Minuit

import matplotlib.pyplot as plt

# Assuming the data is in two columns: channel and counts
# Skip the first 12 lines of the file
data = pd.read_csv('../data/Europium.Spe', skiprows=200, header=None)
untergrund = pd.read_csv('../data/Untergrund.Spe', skiprows=200, header=None)
# Remove the last 14 lines of data
data = data.iloc[:-14]
untergrund = untergrund.iloc[:-14]

# Benenne die Spalte
data.columns = ['Daten']
untergrund.columns = ['Daten']
# Füge einen Index als Channel hinzu
data['Channel'] = data.index
untergrund['Channel'] = untergrund.index

# Sicherstellen, dass die Spalte "data" numerische Werte enthält
data['Daten'] = pd.to_numeric(data['Daten'], errors='coerce')
untergrund['Daten'] = pd.to_numeric(untergrund['Daten'], errors='coerce')

# Normierung des Untergrundes
# Untergrundmessung dauerte 78545s, europiummessung 3718s
untergrund['Daten'] = untergrund['Daten'] * (3718 / 78545)

# Untergrund entfernen
data['Daten'] = data['Daten'] - untergrund['Daten']

# Finde Peaks
peaks_array, properties = find_peaks(data['Daten'], height=10, prominence=20, distance=100)  # Höhe nach Bedarf anpassen
peaks = pd.DataFrame({'peaks': peaks_array, 'peaks_heights': properties['peak_heights']})
peaks = peaks.drop([0,1,2,3])

# Plotten des Spektrums und der Peaks
plt.figure(figsize=(21, 9))

plt.bar(data['Channel'], data['Daten'], linewidth=2, width=1.1, label='Europium')
plt.plot(peaks['peaks'], peaks['peaks_heights'], 'x',color='orange', label='Peaks')
plt.savefig('../plots/Europium.pdf')
plt.clf()

#Literaturwerte einlesen
lit = pd.read_csv('../data/Eu_Lit.csv', skiprows=1, sep='\s+', header=None)
lit.columns = ['Energie', 'Unsicherheit(E)', 'Intensität', 'Unsicherheit(I)']
lit = lit.head(len(peaks))
lit.sort_values(by='Energie', inplace=True, ascending=True)
lit = lit.reset_index(drop=True)

peaks.sort_values(by='peaks', inplace=True, ascending=True)
peaks = peaks.reset_index(drop=True)

peaks = pd.concat([peaks, lit], axis=1)

# Kalibrierung
def kalibrierung(x, a, b):
    return a * x + b

least_squares = LeastSquares(peaks['peaks'], lit['Energie'], lit['Unsicherheit(E)'],kalibrierung)
m = Minuit(least_squares, a=0, b=0)
m.migrad()
m.hesse()

# Plot der Kalibrierung
plt.figure(figsize=(21, 9))
plt.errorbar(peaks['peaks'], peaks['Energie'], yerr=peaks['Unsicherheit(E)'], fmt='o', label='Peaks')
plt.plot(peaks['peaks'], kalibrierung(peaks['peaks'], m.values['a'], m.values['b']), label='Kalibrierung')
fit_info = [f'a = {m.values["a"]:.2f} ± {m.errors["a"]:.2f}', f'b = {m.values["b"]:.2f} ± {m.errors["b"]:.2f}']
plt.xlabel('Channel')
plt.ylabel('Energie [keV]')
plt.legend(title='\n'.join(fit_info), frameon=False)
plt.savefig('../plots/Kalibrierung.pdf')
plt.clf()
