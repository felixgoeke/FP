import numpy as np
import pandas as pd

# Load the peaks positions from peaks.csv
peaks_df = pd.read_csv('./build/peaks.csv')
peak_positions = peaks_df['peaks'].values
print(peak_positions)

# Load the spectrum data from Europium.Spe
spectrum_data = pd.read_csv('./data/Europium.Spe', skiprows=12)
spectrum_data = spectrum_data.iloc[:-14]
spectrum_data.columns = ['Daten']
spectrum_data['Daten'] = pd.to_numeric(spectrum_data['Daten'], errors='coerce')
spectrum_data['index'] = spectrum_data.index

def calculate_line_content(peak_position, spectrum_data, window=5):
    start = max(0, peak_position - window)
    end = min(len(spectrum_data), peak_position + window + 1)
    N = np.sum(spectrum_data['Daten'][start:end])
    N_err = np.sqrt(N)
    return N, N_err

# Calculate the line content and errors for each peak
results = []
for peak in peak_positions:
    N, N_err = calculate_line_content(peak, spectrum_data)
    energy = peaks_df.loc[peaks_df['peaks'] == peak, 'Energie'].values[0]
    intensität = peaks_df.loc[peaks_df['peaks'] == peak, 'Intensität'].values[0]
    uncertainty = peaks_df.loc[peaks_df['peaks'] == peak, 'Unsicherheit(I)'].values[0]
    results.append({'Energie': energy,'Peak Position': peak, 'N': N, 'N_err': N_err, 'Intensität': intensität, 'Unsicherheit(I)': uncertainty})
    #results.append({'Peak Position': peak, 'N': N, 'N_err': N_err})


# Convert results to DataFrame and save to a new CSV file
results_df = pd.DataFrame(results)
results_df.to_csv('./data/line_content_results.csv', index=False)

