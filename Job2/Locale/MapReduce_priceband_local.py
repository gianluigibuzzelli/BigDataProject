import csv
import time
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import re
import pandas as pd
import json  # Per scrivere/leggere i dizionari delle parole
import shutil  # Per pulire le directory temporanee

# --- Impostazioni Dataset e Percorsi ---
datasets = [
    ('/media/gianluigi/Z Slim/DownloadUbuntu/archive/used_cars_data_sampled_1.csv', '10%'),
    ('/media/gianluigi/Z Slim/DownloadUbuntu/archive/used_cars_data_sampled_2.csv', '20%'),
    ('/media/gianluigi/Z Slim/DownloadUbuntu/archive/used_cars_data_sampled_3.csv', '30%'),
    ('/media/gianluigi/Z Slim/DownloadUbuntu/archive/used_cars_data_sampled_4.csv', '40%'),
    ('/media/gianluigi/Z Slim/DownloadUbuntu/archive/used_cars_data_sampled_5.csv', '50%'),
    ('/media/gianluigi/Z Slim/DownloadUbuntu/archive/used_cars_data_sampled_6.csv', '60%'),
    ('/media/gianluigi/Z Slim/DownloadUbuntu/archive/used_cars_data_sampled_7.csv', '70%'),
    ('/media/gianluigi/Z Slim/DownloadUbuntu/archive/used_cars_data_sampled_8.csv', '80%'),
    ('/media/gianluigi/Z Slim/DownloadUbuntu/archive/used_cars_data_sampled_9.csv', '90%'),
    ('/media/gianluigi/Z Slim/DownloadUbuntu/archive/used_cars_data.csv', '100%'),
    ('/media/gianluigi/Z Slim/DownloadUbuntu/archive/used_cars_data_1_5x.csv', '150%'),
]


CHUNK_SIZE = 50_000  

# Colonne essenziali e numeriche (usate per la pulizia dei dati)
essential = ['make_name', 'model_name', 'horsepower', 'engine_displacement']
int_cols = ['daysonmarket', 'dealer_zip', 'listing_id', 'owner_count', 'maximum_seating', 'year']

# Percorsi di output
output_dir = "/media/gianluigi/Z Slim/Risultati_locale/"
os.makedirs(output_dir, exist_ok=True)  

# Nomi dei file di log e grafico
log_file = os.path.join(output_dir, "priceband_mapreduce.txt")
graph_file = os.path.join(output_dir, "priceband_mapreduce.png")

# Directory temporanea per i file di spill delle word_counts
TEMP_WORD_COUNTS_BASE_DIR = os.path.join(output_dir, "temp_model_data_spill")
os.makedirs(TEMP_WORD_COUNTS_BASE_DIR, exist_ok=True)  # Crea la directory base se non esiste

# Variabili per i tempi di esecuzione e le etichette per il grafico
exec_times = []
labels = []

# --- Helper Functions per la Pulizia e l'Estrazione ---
city_pattern = re.compile(r"^[A-Za-zÀ-ÖØ-öø-ÿ'\- ]{1,50}$")
word_pattern = re.compile(r"\b[a-zA-Z]{2,}\b")  # Trova solo parole di almeno 2 caratteri


# --- Map Phase: Processa un Chunk e Spilla le Word Counts su Disco ---
def map_chunk_phase_and_spill(chunk_df, chunk_id, temp_dir):
    stats_chunk = defaultdict(lambda: [0, 0.0])  # key -> [count, sum_days] (rimane in RAM)
    word_counts_chunk = defaultdict(lambda: defaultdict(int))  # key -> {word:count} (verrà spillata)

    for index, row in chunk_df.iterrows():
        try:
            # Pulizia e validazione dei dati
            price_str = str(row.get('price', '0')).replace('$', '').replace(',', '').strip()
            price = float(price_str) if price_str else 0.0

            year_val = str(row.get('year', '0')).strip()
            year = int(float(year_val)) if year_val and year_val.replace('.', '', 1).isdigit() else 0

            city = str(row.get('city', '')).strip()
            days = int(float(row.get('daysonmarket', '0'))) if str(row.get('daysonmarket', '0')).strip().replace('.',
                                                                                                                 '',
                                                                                                                 1).isdigit() else 0

            # Filtri base per dati validi
            if not (0 < price < 1_000_000 and 1900 <= year <= 2025): continue
            if not city_pattern.match(city): continue

            # Determina la fascia di prezzo
            band = 'high' if price > 50000 else 'medium' if price >= 20000 else 'low'
            key = (city, year, band)  # La chiave per l'aggregazione

            # Aggiorna le statistiche (conteggio e somma giorni)
            stats_chunk[key][0] += 1
            stats_chunk[key][1] += days

            # Processa la descrizione per le word counts
            desc = str(row.get('description', '') or '').strip()
            for w in word_pattern.findall(desc.lower()):
                word_counts_chunk[key][w] += 1
        except (ValueError, TypeError):
            # Ignora le righe che causano errori di conversione o dati malformati
            continue

    # SPILL TO DISK: Scrivi i conteggi delle parole di questo chunk su un file JSON temporaneo
    chunk_word_counts_file = os.path.join(temp_dir, f"chunk_{chunk_id}_word_counts.json")
    with open(chunk_word_counts_file, 'w', encoding='utf-8') as f:
        # JSON non supporta tuple come chiavi, quindi convertiamo le chiavi delle tuple in stringhe
        serializable_word_counts = {
            str(k): dict(v) for k, v in word_counts_chunk.items()
        }
        json.dump(serializable_word_counts, f)

    return stats_chunk  # Ritorna solo le statistiche principali per la merge in RAM


# --- Merge Phase: Aggrega le Statistiche Principali in RAM ---
def merge_stats(total_stats, chunk_stats):
    for key, (count, sum_days) in chunk_stats.items():
        total_stats[key][0] += count
        total_stats[key][1] += sum_days
    return total_stats


# --- Shuffle/Merge Phase: Aggrega le Word Counts dai File su Disco ---
def aggregate_word_counts_from_disk(temp_dir):
    final_word_counts = defaultdict(lambda: defaultdict(int))

    json_files = [f for f in os.listdir(temp_dir) if f.endswith('.json')]
    print(f"   --> Aggregazione di {len(json_files)} file di word counts dal disco...")

    for file_name in json_files:
        file_path = os.path.join(temp_dir, file_name)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                chunk_word_counts = json.load(f)
                # Ricostruisci le chiavi tuple dalle stringhe lette dal JSON
                for key_str, words_dict in chunk_word_counts.items():
                    # Assumiamo il formato stringa di tupla: "('city', year, 'band')"
                    # Estrai i componenti rimuovendo le parentesi e dividendo per ", "
                    parts = key_str.strip('()').split(', ')
                    city = parts[0].strip("'")  # Rimuovi gli apici dalla città
                    year = int(parts[1])  # Converti l'anno in int
                    band = parts[2].strip("'")  # Rimuovi gli apici dalla banda di prezzo
                    key = (city, year, band)

                    for word, count in words_dict.items():
                        final_word_counts[key][word] += count
            # Rimuovi il file temporaneo dopo averlo processato per liberare spazio
            os.remove(file_path)
        except Exception as e:
            print(f"   --> Errore nella lettura/aggregazione del file temporaneo {file_path}: {e}")
            continue  # Tenta di continuare con i file rimanenti

    return final_word_counts


# --- Reduce Phase: Calcola i Risultati Finali ---
def reduce_phase(stats, word_counts):
    result = []
    # Ordina le chiavi per garantire un output consistente e riproducibile
    sorted_keys = sorted(stats.keys())
    for key in sorted_keys:
        cnt, sum_days = stats[key]
        city, year, band = key
        # Evita divisione per zero se il conteggio è 0
        avg_days = round(sum_days / cnt, 2) if cnt > 0 else 0.0

        # Trova le top-3 parole per la chiave corrente
        wc = word_counts.get(key, {})  # Usa .get() per gestire chiavi assenti (se un chunk non aveva descrizioni)
        # Ordina le parole per conteggio (decrescente) e poi alfabeticamente (crescente)
        top3 = sorted(wc.items(), key=lambda item: (-item[1], item[0]))[:3]
        top3_words = [word for word, count in top3]  # Estrai solo le parole

        result.append((city, year, band, cnt, avg_days, top3_words))
    # Ordina il risultato finale per coerenza
    return sorted(result)


# --- Main Execution Loop ---
with open(log_file, 'w', encoding='utf-8') as out:
    for path, label in datasets:
        out.write(f"\n== Dataset {label}: {path} ==\n")
        print(f"\nProcessing {label}...", end=' ')

        # Verifica se il file del dataset esiste
        if not os.path.exists(path):
            print(f"ATTENZIONE: Il file non esiste al percorso: {path}. Saltando questo dataset.")
            out.write(f"ATTENZIONE: Il file non esiste al percorso: {path}. Saltando questo dataset.\n")
            continue

        # Crea e pulisci la directory temporanea specifica per il dataset corrente
        current_temp_word_counts_dir = os.path.join(TEMP_WORD_COUNTS_BASE_DIR,
                                                    label.replace('%', '_').replace('.', '_'))
        os.makedirs(current_temp_word_counts_dir, exist_ok=True)
        # Rimuovi tutti i file temporanei da una precedente esecuzione per questo dataset
        for f_name in os.listdir(current_temp_word_counts_dir):
            try:
                os.remove(os.path.join(current_temp_word_counts_dir, f_name))
            except OSError as e:
                print(f"   --> Errore durante la pulizia del file temporaneo {f_name}: {e}")

        start_time = time.time()

        # Inizializza le statistiche totali che rimarranno in RAM
        total_stats = defaultdict(lambda: [0, 0.0])

        chunk_counter = 0
        try:
            # Leggi il CSV in chunk con pandas
            for chunk_counter, chunk_df in enumerate(
                    pd.read_csv(path, chunksize=CHUNK_SIZE, encoding='utf-8', on_bad_lines='skip', dtype=str)):
                print(f"   --> Processing chunk {chunk_counter + 1} ({len(chunk_df)} rows)...")

                # Esegui la fase di mappatura e spilla le word counts su disco
                stats_chunk = map_chunk_phase_and_spill(chunk_df, chunk_counter, current_temp_word_counts_dir)

                # Unisci le statistiche del chunk ai totali globali in RAM
                total_stats = merge_stats(total_stats, stats_chunk)

            # Fase di Shuffle/Merge: Aggrega tutte le word counts dai file su disco
            final_word_counts = aggregate_word_counts_from_disk(current_temp_word_counts_dir)

            # Esegui la fase di Riduzione con i dati finali
            res = reduce_phase(total_stats, final_word_counts)

            duration = round(time.time() - start_time, 2)

            # Stampa i primi 10 risultati nel log e a console
            for rec in res[:10]:
                line = f"{rec[0]} | {rec[1]} | {rec[2]} | num={rec[3]} | avg_days={rec[4]} | top3={rec[5]}"
                out.write(line + "\n")
            print(f"   --> Tempo totale per {label}: {duration} sec")
            out.write(f"Tempo totale per {label}: {duration} sec\n")
            exec_times.append(duration)
            labels.append(label)

        except Exception as e:
            print(f"Errore critico durante l'elaborazione del dataset {label} al chunk {chunk_counter + 1}: {e}")
            out.write(f"Errore critico durante l'elaborazione del dataset {label} al chunk {chunk_counter + 1}: {e}\n")
            duration = round(time.time() - start_time, 2)
            exec_times.append(duration)
            labels.append(label + " (Errore)")  # Segnala l'errore nel grafico
            continue  # Passa al prossimo dataset
        finally:
            # Pulizia finale della directory temporanea specifica del dataset
            if os.path.exists(current_temp_word_counts_dir):
                try:
                    shutil.rmtree(current_temp_word_counts_dir)
                    print(f"   --> Pulita directory temporanea per {label}: {current_temp_word_counts_dir}")
                except OSError as e:
                    print(f"   --> Errore durante la pulizia finale di {current_temp_word_counts_dir}: {e}")

# --- Plot dei Risultati ---
plt.figure(figsize=(8, 5))  # Aumentato dimensione grafico
plt.plot(labels, exec_times, marker='o', linestyle='-', color='b')
plt.title("MapReduce Locale City–Year–PriceBand (Spill to Disk)", fontsize=16)
plt.xlabel("Dataset %", fontsize=12)
plt.ylabel("Tempo (s)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45, ha='right', fontsize=10)  # Ruota e dimensiona le etichette
plt.yticks(fontsize=10)
plt.tight_layout()  # Adatta i margini
plt.savefig(graph_file)
plt.close()

print("\n--- Processo Completato ---")
print(f"Report dettagliato salvato in: {log_file}")
print(f"Grafico dei tempi di esecuzione salvato in: {graph_file}")
