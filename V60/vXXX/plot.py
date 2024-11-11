import matplotlib.pyplot as plt
import numpy as np
from uncertainties import ufloat
import uncertainties.unumpy as unp
import pandas as pd
import scipy.constants as const
import math

# Diese Funktion liest Daten aus einer CSV-Datei ein und gibt sie als Numpy-Array zurück.
def Data(Name):
    # Daten aus der CSV-Datei lesen und als DataFrame speichern
    Daten1 = pd.read_csv(Name, skiprows=0, sep=";")
    # Kommas in den Daten durch Punkte ersetzen
    Daten2 = Daten1.replace(",", ".", regex=True)
    # DataFrame in ein Numpy-Array konvertieren und den Datentyp auf float64 festlegen
    return Daten2.to_numpy(dtype=np.float64)


# Diese Funktion berechnet den größten Exponenten für jedes Element in einem 2D-Array.
def max_exponent_2d(array):
    max_exp = []  # Liste zur Speicherung der größten Exponenten
    for i in range(len(array[0])):  # Iteration über die Spalten des Arrays
        max_exp.append(int(np.floor(np.log10(np.abs(array[:,i].max())))))  # Berechnung des größten Exponenten für die aktuelle Spalte
    return max_exp  # Rückgabe der Liste der größten Exponenten


# Diese Funktion berechnet den kleinsten Exponenten für jedes Element in einem 2D-Array.
def min_exponent_2d(array):
    min_exp = []  # Liste zur Speicherung der kleinsten Exponenten
    for i in range(len(array[0])):  # Iteration über die Spalten des Arrays
        non_zero_values = array[:,i][np.nonzero(array[:,i])]  # Auswahl der nicht-null Werte in der aktuellen Spalte
        if len(non_zero_values) == 0:  # Überprüfung, ob es keine nicht-null Werte gibt
            min_exp.append(0)  # Wenn keine nicht-null Werte vorhanden sind, wird der Exponent als 0 gesetzt
        else:
            min_exp.append(int(np.floor(np.log10(np.abs(abs(non_zero_values).min())))))  # Berechnung des kleinsten Exponenten für die nicht-null Werte in der aktuellen Spalte
    return min_exp  # Rückgabe der Liste der kleinsten Exponenten


# Diese Funktion konvertiert ein 2D-Array in eine LaTeX-Tabelle und speichert sie in einer Datei.
# Diese Funktion konvertiert ein 2D-Array in eine LaTeX-Tabelle und speichert sie in einer Datei.
def array_to_latex_table(array, filename):
    exponent=max_exponent_2d(array)  # Berechnung der größten Exponenten für jedes Element im Array
    minexponent=min_exponent_2d(array)  # Berechnung der kleinsten Exponenten für jedes Element im Array
    with open(filename, "w") as f:  # Öffnen der Datei im Schreibmodus
        for row in array:  # Iteration über die Zeilen des Arrays
            formatted_row = []  # Liste zur Speicherung der formatierten Zeile
            i=0
            for cell in row:  # Iteration über die Zellen der aktuellen Zeile
                if (isinstance(cell, int) or (isinstance(cell, float) and cell.is_integer())) and exponent[i] <=5 :
                    # Überprüfung, ob die Zelle ein ganzzahliger Wert ist und der Exponent kleiner oder gleich 5 ist
                    formatted_row.append("{:.0f}".format(cell))  # Formatierung der Zelle als ganze Zahl
                elif exponent[i] < -2:  # Überprüfung, ob der Exponent kleiner als -2 ist
                    formatted_row.append("{:.2f}".format(cell * 10**-minexponent[i], minexponent[i]).replace(".", ","))
                    # Formatierung der Zelle mit Exponentennotation und Anpassung des Dezimaltrennzeichens
                elif exponent[i] >5:  # Überprüfung, ob der Exponent größer als 5 ist
                    formatted_row.append("{:.2f}".format(cell * 10**-minexponent[i], minexponent[i]).replace(".", ","))
                    # Formatierung der Zelle mit Exponentennotation und Anpassung des Dezimaltrennzeichens
                elif (10*cell).is_integer():  # Überprüfung, ob das Zehnfache der Zelle eine ganze Zahl ist
                    formatted_row.append("{:.1f}".format(cell).replace(".", ","))
                    # Formatierung der Zelle mit einer Dezimalstelle und Anpassung des Dezimaltrennzeichens
                else:
                    formatted_row.append("{:.2f}".format(cell).replace(".", ","))
                    # Formatierung der Zelle mit zwei Dezimalstellen und Anpassung des Dezimaltrennzeichens
                i=i+1
            f.write(" & ".join(formatted_row))  # Zusammenfügen der formatierten Zellen mit dem Trennzeichen "&"
            f.write(" \\\\\n")  # Schreiben der formatierten Zeile in die Datei mit einem Zeilenumbruch
    return minexponent  # Rückgabe der Liste der kleinsten Exponenten


# Diese Funktion konvertiert ein 1D-Array in eine LaTeX-Tabelle und speichert sie in einer Datei.
def array_to_latex_table_1D(array, filename):
    exponent=max_exponent_2d(array)
    minexponent=min_exponent_2d(array)
    with open(filename, "w") as f:
        formatted_array = []
        i=0
        for cell in array:
            formatted_array=[]
            if (isinstance(cell, int) or (isinstance(cell, float) and cell.is_integer())) and exponent[i] <= 5:
                formatted_array.append("{:.0f}".format(cell))
            elif exponent[i] < -2:
                formatted_array.append("{:.2f}".format(cell * 10**-minexponent[i], -minexponent[i]).replace(".", ","))
            elif exponent[i] >= 5:
                formatted_array.append("{:.2f}".format(cell * 10**-minexponent[i], -minexponent[i]).replace(".", ","))
            else:
                formatted_array.append("{:.2f}".format(cell).replace(".", ","))
           
            f.write(", ".join(formatted_array))
            f.write(" \\\\\n")
    return minexponent

# Diese Funktion konvertiert ein 3D-Array in eine LaTeX-Tabelle und speichert sie in einer Datei.
def array_to_latex_table_3d(array, filename):
    exponent=max_exponent_2d(array[:,:,0].T)
    minexponent=min_exponent_2d(array[:,:,0].T)
    with open(filename, "w") as f:
        for row in zip(*array):
            formatted_row = []
            i=0
            for cell in row:
                if np.isnan(cell[0]):
                    formatted_row.append("")
                else:
                    if (isinstance(cell[0], int) or (isinstance(cell[0], float) and cell[0].is_integer())) and exponent[i] <= 5:
                        formatted_row.append("${:.0f} \\pm {:.0f}$".format(cell[0], cell[1]))
                    elif exponent[i] < -2:
                        formatted_row.append("${:.2f} \\pm {:.2f}$".format(cell[0] * 10**-minexponent[i], cell[1] * 10**-minexponent[i], minexponent[i]).replace(".", ","))
                    elif exponent[i] >= 5:
                        formatted_row.append("${:.2f} \\pm {:.2f}$".format(cell[0] * 10**-minexponent[i], cell[1] * 10**-minexponent[i], minexponent[i]).replace(".", ","))
                    elif (10*cell[0]).is_integer():
                            formatted_row.append("${:.1f}\\pm{:.1f}".format(cell).replace(".", ","))
                    else:
                        formatted_row.append("${:.2f} \\pm {:.2f}$".format(cell[0], cell[1]).replace(".", ","))
                i=i+1
            f.write(" & ".join(formatted_row))
            f.write(" \\\\\n")
    return minexponent

# Diese Funktion berechnet den Mittelwert und die Standardabweichung eines Datensatzes.
def Datapoint(Daten):
    mean = np.mean(Daten)
    std = np.std(Daten, ddof=1) / np.sqrt(len(Daten))
    return mean, std
# Diese Funktion führt eine lineare Regression durch und gibt die Parameter und Fehler aus.

# Funktion zur linearen Regression mit Gewichtung der Daten
def linear_regression(x, y, color):
    # Berechnung der Regressionsparameter und Kovarianzmatrix
    params, covariance_matrix = np.polyfit(x, unp.nominal_values(y), w=1/unp.std_devs(y), deg=1, cov=True)

    # Berechnung der Fehler
    errors = np.sqrt(np.diag(covariance_matrix))

    # Ausgabe der Parameter und Fehler
    for name, value, error in zip('ab', params, errors):
        print(f'{name}_{color} = {value:.3g} ± {error:.3g}')
    
    return params, errors

# Funktion zur linearen Regression ohne Gewichtung der Daten
def linear_regression2(x, y, color):
    # Berechnung der Regressionsparameter und Kovarianzmatrix
    params, covariance_matrix = np.polyfit(x, unp.nominal_values(y), deg=1, cov=True)

    # Berechnung der Fehler
    errors = np.sqrt(np.diag(covariance_matrix))

    # Ausgabe der Parameter und Fehler
    for name, value, error in zip('ab', params, errors):
        print(f'{name}_{color} = {value:.3g} ± {error:.3g}')
    
    return params, errors

# Funktion zum Erstellen eines einfachen Plots
def Plot1(x, y, xlabel="", ylabel="", filepath=""):
    fig, ax = plt.subplots()
    ax.plot(x, y, "r.", markersize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.grid()
    plt.savefig(filepath)
    plt.clf()

# Funktion zum Erstellen eines Plots mit zwei Datensätzen
def Plot2(x, y1, y2, xlabel="", ylabel="", filepath="", label1="", label2=""):
    fig, ax = plt.subplots()
    ax.plot(x, y1, "r.", markersize=8, label=label1)
    ax.plot(x, y2, "b.", markersize=8, label=label2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.savefig(filepath)
    plt.clf()

# Funktion zum Erstellen eines Plots mit Fehlerbalken und optionaler Maskierung
def Plot3(high, x, y, xlabel="", y1label="", label="", filepath="", mask=[]):
    fig, ax1 = plt.subplots()
    fig.tight_layout()
    fig.subplots_adjust(right=0.8)
    ax1.errorbar(x[mask], unp.nominal_values(y[mask]), yerr=unp.std_devs(y[mask]), xerr=None, color="r", fmt=".", label=label)
    ax1.errorbar(x[~mask], unp.nominal_values(y[~mask]), yerr=unp.std_devs(y[~mask]), xerr=None, color="k", fmt=".", label="Messpunkte ohne Aussagekraft")
    if high:
        ax1.errorbar(x[high-1], unp.nominal_values(y[high-1]), yerr=unp.std_devs(y[high-1]), xerr=None, color="g", fmt=".", label="Beginn der linearen Regression des langen Zerfalls")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(y1label)
    ax1.grid()
    return fig, ax1, mask

# Funktion zum Hinzufügen einer Linie zu einem vorhandenen Plot
def Plot4(fig, ax, x, y, label1="", xlim1=[], ylim=[], color=""):
    ax.plot(x, y, label=label1, color=color)
    ax.set_xlim(xlim1)
    ax.set_ylim(ylim)




#Funktion zum Erstellen von Latex Tabellen aus python Arrays
#exp1=array_to_latex_table(newarray, "content/Tabelle1.tex") 

