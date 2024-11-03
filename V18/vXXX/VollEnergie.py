import numpy as np
import pandas as pd
from uncertainties.umath import *
from uncertainties import ufloat
from datetime import datetime
import iminuit
from iminuit import Minuit
from iminuit.cost import LeastSquares
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.ticker as ticker
import os

start_aktivität = ufloat(4130, 60)  # Bq, am 1.10.2000

start_date = datetime(2000, 10, 1)
end_date = datetime(2024, 10, 21)

# Differenz zwischen den Daten
time_difference = end_date - start_date

# Umwandlung der Zeitdifferenz in Sekunden
time_difference_in_seconds = time_difference.total_seconds()
print(f"Die Zeitdifferenz beträgt: {time_difference_in_seconds} s")

halbwertszeit_eu = (13.522) * 365.25 * 24 * 60 * 60  # Jahre, in s umgerechnet
print(f"Die Halbwertszeit beträgt: {halbwertszeit_eu} s")

def aktivitätsgesetz(t, A0, tau):
    """Formel für die Aktivität einer Probe"""
    A = A0 * np.exp(((-np.log(2)) / tau) * t)
    return A


end_aktivität = aktivitätsgesetz(
    time_difference_in_seconds, start_aktivität, halbwertszeit_eu
)
print(f"Die Endaktivität beträgt: {end_aktivität} Becquerel")
a = 7.02 + 1.5  # cm
r = 2.25  # cm
omega_4pi = 1 / 2 * (1 - a / (np.sqrt(a**2 + r**2)))
print(f"Der Raumwinkel Omega beträgt {omega_4pi:.5f}")

peaks = pd.read_csv("./data/line_content_results.csv")


def fedp(omega, N, A, W, t):
    """Formel für die Vollenergienachweiswahrscheinlichkeit"""
    Q = (4 * np.pi * N) / (omega * A * W * t)
    return Q


# Messzeit
t = 3634  # s

# Berechnung der Vollenergienachweiswahrscheinlichkeit für die Peaks
results = []

for index, row in peaks.iloc[3:].iterrows():
    N = ufloat(row['N'], row['N_err'])
    W = ufloat(row['Intensität'], row['Unsicherheit(I)'])  # Annahme: 'Intensität_err' ist die Spalte für die Unsicherheit in peaks.csv
    Q = fedp(omega_4pi, N, end_aktivität, W, t)
    results.append([row['Energie'], N.nominal_value, N.std_dev, W.nominal_value, W.std_dev, Q.nominal_value, Q.std_dev])

# Speichern der Ergebnisse in einer neuen Textdatei
#results_df = pd.DataFrame(results, columns=['Energie', 'N', 'N_err', 'Intensität', 'Intensität_err', 'Q', 'Q_err'])
#results_df.to_csv('./build/peaks_result.txt', index=False, sep='\t')
# Speichern der Ergebnisse in einer neuen Textdatei und Runden der Ergebnisse auf 5 Nachkommastellen
results_df = pd.DataFrame(results, columns=['Energie', 'N', 'N_err', 'Intensität', 'Intensität_err', 'Q', 'Q_err'])
results_df['Q'] = results_df['Q'].round(5)
results_df['Q_err'] = results_df['Q_err'].round(5)
results_df.to_csv('./build/vollenergie_results.csv', index=False, sep='\t')
print("Die Berechnungen wurden abgeschlossen und in 'peaks_result.txt' gespeichert.")

def q_energy(e, a, b):
    """Formel für die Energieabhängigkeit der Vollenergienachweiswahrscheinlichkeit"""
    Q = a * e** b 
    return Q

# Daten für den Fit vorbereiten
x_data = results_df['Energie']
y_data = results_df['Q']
y_err = results_df['Q_err']

# LeastSquares-Kostenfunktion erstellen
least_squares = LeastSquares(x_data, y_data, y_err, q_energy)

# Minuit-Objekt erstellen und den Fit durchführen
m = Minuit(least_squares, a=1, b=-0.5)
m.migrad()

# Fit-Ergebnisse mit Fehlern ausgeben
a_value = m.values['a']
a_error = m.errors['a']
b_value = m.values['b']
b_error = m.errors['b']
print(m)
print(f"Fit-Ergebnisse: a = {a_value} ± {a_error}, b = {b_value} ± {b_error}")

# Fit-Kurve plotten
x_fit = np.linspace(min(x_data) - 10, max(x_data) + 10, 500)
y_fit = q_energy(x_fit, m.values['a'], m.values['b'])
plt.figure(figsize=(10, 6))
plt.plot(x_fit, y_fit, label='Fit', color='blue')

# Plotten der Energie gegen Q
plt.errorbar(results_df['Energie'], results_df['Q'], yerr=results_df['Q_err'], fmt='o', ecolor='r', capsize=5, label='Messdaten')
plt.xlabel('Energie (keV)')
plt.ylabel('Vollenergienachweiswahrscheinlichkeit Q')
plt.legend()
plt.grid(True)
plt.legend(title=f'Fit: a={m.values["a"]:.5f}, b={m.values["b"]:.5f}')
plt.savefig('./plots/energie_vs_Q.pdf')
print("Das Diagramm wurde als 'energie_vs_Q.pdf' gespeichert.")

## Fit der Energieabhängigkeit der Vollenergienachweiswahrscheinlichkeit
## Daten für den Fit vorbereiten
#x_data = results_df['Energie']
#y_data = results_df['Q']
#y_err = results_df['Q_err']
#
## LeastSquares-Kostenfunktion erstellen
#least_squares = LeastSquares(x_data, y_data, y_err, q_energy)
#
## Minuit-Objekt erstellen und den Fit durchführen
#m = Minuit(least_squares, a=1, b=-0.5,c)
#m.migrad()
#
#
## Fit-Ergebnisse mit Fehlern ausgeben
#
#a_value = m.values['a']
#a_error = m.errors['a']
#b_value = m.values['b']
#b_error = m.errors['b']
#print(m)
#print(f"Fit-Ergebnisse: a = {a_value} ± {a_error}, b = {b_value} ± {b_error}")
## Fit-Kurve plotten
#x_fit = np.linspace(min(x_data-10), max(x_data+10), 500)
#y_fit = q_energy(x_fit, m.values['a'], m.values['b'])
#plt.figure(figsize=(10, 6))
#plt.plot(x_fit, y_fit, label='Fit', color='blue')
#
## Plotten der Energie gegen Q
#plt.errorbar(results_df['Energie'], results_df['Q'], yerr=results_df['Q_err'], fmt='o', ecolor='r', capsize=5, label='Messdaten')
#plt.xlabel('Energie (keV)')
#plt.ylabel('Vollenergienachweiswahrscheinlichkeit Q')
##plt.title('Energie vs. Vollenergienachweiswahrscheinlichkeit')
#plt.legend()
#plt.grid(True)
#plt.legend(title=f'Fit: a={m.values["a"]:.5f}, b={m.values["b"]:.5f}')
#plt.savefig('./plots/energie_vs_Q.pdf')
#print("Das Diagramm wurde als 'energie_vs_Q.pdf' gespeichert.")
