import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker # Importiamo il modulo ticker
import os # Importiamo il modulo os per la gestione dei percorsi

# Funzione per convertire le etichette percentuali in valori numerici
def parse_percentage_labels(labels):
    """Converte una lista di stringhe percentuali ('X%') in un array di float (0.X)."""
    return np.array([float(label.replace('%', '')) / 100.0 for label in labels])

# Dati del primo set: Spark core (Locale)
labels_set1 = ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%', '200%', '400%']
exec_times_set1 = np.array([5.98, 10.36, 14.0, 17.54, 21.76, 26.33, 31.29, 36.03, 40.15, 47.54, 101.44, 503.21])

# Dati del secondo set: Spark core (Cluster)
labels_set2 = ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%', '200%', '400%']
exec_times_set2 = np.array([8.57, 12.23, 15.5, 18.09, 20.03, 22.76, 24.35, 26.21, 28.02, 30.0, 55.12, 105.20])

# Convertiamo le etichette in valori numerici per l'asse X
x_values_set1 = parse_percentage_labels(labels_set1)
x_values_set2 = parse_percentage_labels(labels_set2)

# Uniamo tutte le etichette uniche per l'asse X e le ordiniamo per visualizzazione
all_numeric_x_values = np.unique(np.concatenate((x_values_set1, x_values_set2)))
# Convertiamo i valori numerici unici in stringhe percentuali per le etichette dei tick
all_x_labels = [f"{int(val * 100)}%" for val in all_numeric_x_values]

# Creazione della figura e degli assi del plot con le dimensioni specificate (8x5 pollici).
plt.figure(figsize=(8, 5))

# Plot del primo set di dati (Locale)
plt.plot(x_values_set1, exec_times_set1, marker='o', linestyle='-', color='blue', label='Locale')

# Plot del secondo set di dati (Cluster)
plt.plot(x_values_set2, exec_times_set2, marker='s', linestyle='--', color='red', label='Cluster') # Marker quadrato e linea tratteggiata per distinguere

# Aggiunta di titolo ed etichette agli assi
plt.title("Confronto Tempi di Esecuzione Job 1 Spark Core (Locale vs Cluster)", fontsize=16)
plt.xlabel("Sample size", fontsize=13)
plt.ylabel("Execution time (s)", fontsize=13)

# Configurazione delle etichette dell'asse X per mostrare le percentuali
# Manteniamo le etichette originali sui tick specificati
plt.xticks(all_numeric_x_values, all_x_labels, rotation=45, ha='right', fontsize=10)

# Impostazione del limite dell'asse Y. Consideriamo il massimo di entrambi i set di dati.
max_overall_time = max(np.max(exec_times_set1), np.max(exec_times_set2))
plt.ylim(0, max_overall_time + 10) # Aggiungiamo un margine per chiarezza

# --- Modifiche per una griglia uniforme ---
# Abilitiamo i minor ticks
plt.minorticks_on()

# Definiamo i locatori per le tacche principali e secondarie sull'asse X
# Tacche principali ogni 0.2 (20%), tacche secondarie ogni 0.1 (10%)
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(0.2))
plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(0.1))

# Definiamo i locatori per le tacche principali e secondarie sull'asse Y
# Tacche principali ogni 50 unità, tacche secondarie ogni 25 unità
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(50))
plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(25))

# Aggiungiamo la griglia principale (major grid)
plt.grid(True, which='major', linestyle='-', linewidth='0.7', color='gray', alpha=0.7)

# Aggiungiamo la griglia secondaria (minor grid)
# Avrà uno stile diverso (linee tratteggiate) e sarà più chiara per distinguere
plt.grid(True, which='minor', linestyle=':', linewidth='0.5', color='lightgray', alpha=0.5)
# --- Fine modifiche griglia ---

# Aggiunta della legenda per identificare le due linee
plt.legend(fontsize=12)

# Ottimizzazione del layout
plt.tight_layout()

# --- Parte aggiunta per salvare il plot ---
# Definisci il percorso e il nome del file dove salvare il grafico.
# Questo percorso è per l'esecuzione locale sul tuo computer.
output_directory = '/home/gianluigi/CodiciBigData'
output_filename = 'confronto_tempi_spark_core.png'
full_path = os.path.join(output_directory, output_filename)

# Crea la directory di output se non esiste
os.makedirs(output_directory, exist_ok=True)

# Salva il plot nel file specificato
plt.savefig(full_path)
print(f"Grafico salvato in: {full_path}")
# --- Fine parte aggiunta ---

# Mostra il plot (opzionale, se vuoi vederlo anche a schermo)
plt.show()
