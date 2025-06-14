import csv
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import gc  # Importa il modulo garbage collector
import pandas as pd  # Necessario per la lettura a chunk
import json  # Necessario per lo spill su disco
import shutil  # Necessario per pulire le directory temporanee
from collections import defaultdict

# --- 1) Configurazione input/output ---
# Percorsi dei dataset
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
    ('/media/gianluigi/Z Slim/DownloadUbuntu/archive/used_cars_data_2x.csv', '200%'),
    ('/media/gianluigi/Z Slim/DownloadUbuntu/archive/used_cars_data_4x.csv', '400%')
]

# Percorsi di output per report e grafico
output_dir = "/media/gianluigi/Z Slim/Risultati_locale/"
os.makedirs(output_dir, exist_ok=True)
log_file = os.path.join(output_dir, "model_similarity_local_spill2.txt")
graph_file = os.path.join(output_dir, "model_similarity_local_spill_times.png")

# Directory temporanea per i file di spill dei risultati intermedi per modello
TEMP_MODEL_DATA_DIR = os.path.join(output_dir, "temp_model_data_spill")
os.makedirs(TEMP_MODEL_DATA_DIR, exist_ok=True)  # Crea la directory base se non esiste

# Dimensione del chunk per la lettura del CSV.
# Questo valore è cruciale per il consumo di RAM. SPERIMENTA!
CHUNK_SIZE = 50_000  # Ho ridotto il CHUNK_SIZE per maggiore sicurezza

exec_times = []
labels = []


# --- Funzioni ausiliarie per MapReduce con Spill ---

def process_chunk_and_spill(chunk_df, chunk_id, temp_dir):
    """
    Map Phase: Reads a chunk, cleans, filters, and prepares data for aggregation.
    Emits (model_name, (hp, ed, price, count=1)) and spills to disk.
    Also returns a local dictionary for prices and HPs for the current chunk.
    """
    chunk_aggregated_data = defaultdict(lambda: [0.0, 0.0, 0.0, 0])  # [hp, ed, price, count]
    chunk_prices_lookup = {}
    chunk_hps_lookup = {}

    for index, row in chunk_df.iterrows():
        try:
            # Pulizia e validazione dei dati
            price_str = str(row.get('price', '')).replace('$', '').replace(',', '').strip()
            horsepower_str = str(row.get('horsepower', '')).strip()
            engine_displacement_str = str(row.get('engine_displacement', '')).strip()
            model_name = str(row.get('model_name', '')).strip()  # Assicurati che sia una stringa non vuota

            price = float(price_str) if price_str else 0.0
            horsepower = float(horsepower_str) if horsepower_str else 0.0
            engine_displacement = float(engine_displacement_str) if engine_displacement_str else 0.0

            # Filtra out invalid entries e model_name vuoto
            if (model_name and
                    horsepower > 0 and
                    engine_displacement > 0 and
                    price > 0):
                # Accumula per l'aggregazione di questo chunk
                chunk_aggregated_data[model_name][0] += horsepower
                chunk_aggregated_data[model_name][1] += engine_displacement
                chunk_aggregated_data[model_name][2] += price  # Accumuilamo il prezzo anche qui
                chunk_aggregated_data[model_name][3] += 1

                # Manteniamo lookup per i dati individuali per il join finale (se necessario, altrimenti si spilla)
                # Per questo codice, non usiamo questi lookup del CHUNK
                # ma serve per dimostrare l'idea se volessimo spillare anche questi.

        except (ValueError, TypeError):
            continue  # Salta righe con errori di parsing

    # SPILL TO DISK: Scriviamo i dati aggregati del chunk per modello
    # {model_name: [sum_hp, sum_ed, sum_price, count]}
    chunk_spill_file = os.path.join(temp_dir, f"chunk_{chunk_id}_model_data.json")
    with open(chunk_spill_file, 'w', encoding='utf-8') as f:
        # Converti defaultdict in dict standard per la serializzazione JSON
        json.dump(dict(chunk_aggregated_data), f)

    return  # Non ritorniamo nulla, i dati sono spillati


def aggregate_model_data_from_disk(temp_dir):
    """
    Reduce Phase (Shuffle/Merge): Aggregates model features from all spilled files.
    Returns {model_name: (avg_hp, avg_ed), model_name: avg_price, model_name: avg_hp}.
    """
    total_aggregated = defaultdict(lambda: [0.0, 0.0, 0.0, 0])  # [sum_hp, sum_ed, sum_price, count]

    json_files = [f for f in os.listdir(temp_dir) if f.endswith('.json')]
    print(f"   --> Aggregazione di {len(json_files)} file di dati intermedi per modello dal disco...")

    for file_name in json_files:
        file_path = os.path.join(temp_dir, file_name)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                chunk_data = json.load(f)
                for model_name, (hp_sum, ed_sum, price_sum, count) in chunk_data.items():
                    total_aggregated[model_name][0] += hp_sum
                    total_aggregated[model_name][1] += ed_sum
                    total_aggregated[model_name][2] += price_sum
                    total_aggregated[model_name][3] += count
            os.remove(file_path)  # Pulisci il file temporaneo dopo la lettura
        except Exception as e:
            print(f"   --> Errore nella lettura/aggregazione del file {file_path}: {e}")
            continue

    # Calcola le medie finali e prepara i dizionari per le fasi successive
    model_features_dict = {}  # {model_name: (avg_hp, avg_ed)}
    prices_dict = {}  # {model_name: avg_price}
    hps_dict = {}  # {model_name: avg_hp} (per top_power_model)

    for model_name, (hp_sum, ed_sum, price_sum, count) in total_aggregated.items():
        if count > 0:
            avg_hp = round(hp_sum / count, 2)
            avg_ed = round(ed_sum / count, 2)
            avg_price = round(price_sum / count, 2)

            model_features_dict[model_name] = (avg_hp, avg_ed)
            prices_dict[model_name] = avg_price
            hps_dict[model_name] = avg_hp  # Usiamo la media per questo calcolo

    return model_features_dict, prices_dict, hps_dict


def generate_similar_pairs(model_features_dict):
    """
    Generates similar model pairs based on horsepower and engine displacement.
    Returns a set of unique sorted tuples of (model1, model2).
    This phase still needs to load all model_features_dict in RAM.
    """
    pairs = set()
    model_items = list(model_features_dict.items())  # Convert to list to iterate multiple times
    num_models = len(model_items)

    print(f"   --> Generazione coppie simili per {num_models} modelli unici. Questo può richiedere tempo/RAM (O(M^2)).")
    for i in range(num_models):
        m1, (hp1, ed1) = model_items[i]
        for j in range(i, num_models):  # Avoid duplicates (m1, m2) and (m2, m1)
            m2, (hp2, ed2) = model_items[j]
            if m1 == m2:  # Skip self-comparison
                continue

            # Check similarity condition (handle division by zero for hp1/ed1 if they are 0)
            if hp1 > 0 and ed1 > 0:  # Ensure denominator is not zero
                if (abs(hp1 - hp2) / hp1 <= 0.10 and abs(ed1 - ed2) / ed1 <= 0.10):
                    pairs.add(tuple(sorted((m1, m2))))  # Ensure consistent order for uniqueness
    return pairs


def map_explode_groups(groups_dict):
    """
    Map Phase: Transforms groups into (member, group_name) for joins.
    Uses a generator.
    """
    for group_name, members in groups_dict.items():
        for member in members:
            yield (member, group_name)


def reduce_avg_price_by_group(exploded_data, prices_dict):
    """
    Reduce Phase: Calculates average price for each group.
    Returns a dictionary {group_name: avg_price}.
    """
    group_prices = {}  # {group_name: [list_of_prices]}
    for member, group_name in exploded_data:
        price = prices_dict.get(member)
        if price is not None:  # Ensure price exists for the member
            group_prices.setdefault(group_name, []).append(price)

    result = {
        group_name: round(sum(prices) / len(prices), 2)
        for group_name, prices in group_prices.items() if prices  # Ensure no division by zero
    }
    return result


def reduce_top_power_model_by_group(exploded_data, hps_dict):
    """
    Reduce Phase: Finds the model with the highest horsepower in each group.
    Returns a dictionary {group_name: top_power_model}.
    """
    group_hps = {}  # {group_name: (current_top_model, current_max_hp)}
    for member, group_name in exploded_data:
        hp = hps_dict.get(member)
        if hp is not None:  # Ensure horsepower exists for the member
            current_top_model, current_max_hp = group_hps.get(group_name, (None, -1))
            if hp > current_max_hp:
                group_hps[group_name] = (member, hp)

    result = {
        group_name: top_model
        for group_name, (top_model, _) in group_hps.items() if top_model  # Ensure a top model was found
    }
    return result


# --- Main Execution Loop ---
with open(log_file, "w", encoding="utf-8") as fout:
    for path, label in datasets:
        fout.write(f"\n== Dataset {label}: {path} ==\n")
        print(f"\n== Dataset {label}: {path} ==")

        # Crea e pulisci la directory temporanea specifica per il dataset corrente
        # Usiamo il label per il nome della directory (es. "10%")
        # Rimuovi i caratteri speciali che potrebbero causare problemi nel nome della directory
        safe_label = label.replace('%', 'p').replace('.', '_')
        current_temp_model_data_dir = os.path.join(TEMP_MODEL_DATA_DIR, safe_label)
        os.makedirs(current_temp_model_data_dir, exist_ok=True)
        # Rimuovi tutti i file temporanei da una precedente esecuzione per questo dataset
        for f_name in os.listdir(current_temp_model_data_dir):
            try:
                os.remove(os.path.join(current_temp_model_data_dir, f_name))
            except OSError as e:
                print(f"   --> Errore durante la pulizia del file temporaneo {f_name}: {e}")

        start_time_total = time.time()

        # ---- 2) Processa i dati in chunk e spilla le aggregazioni per modello ----
        # Questa è la fase di Map iniziale, che aggrega parzialmente e spilla.
        print("   Fase 1: Processamento chunk e spill su disco...")
        chunk_counter = 0
        try:
            for chunk_counter, chunk_df in enumerate(
                    pd.read_csv(path, chunksize=CHUNK_SIZE, encoding='utf-8', on_bad_lines='skip', dtype=str)):
                print(f"      --> Processing chunk {chunk_counter + 1} ({len(chunk_df)} rows)...")
                process_chunk_and_spill(chunk_df, chunk_counter, current_temp_model_data_dir)

            # Forzo il garbage collection dopo aver processato tutti i chunk per liberare memoria
            del chunk_df  # Elimino il riferimento all'ultimo chunk_df
            gc.collect()

        except Exception as e:
            print(
                f"Errore critico durante la fase di Map/Spill per il dataset {label} al chunk {chunk_counter + 1}: {e}")
            fout.write(
                f"Errore critico durante la fase di Map/Spill per il dataset {label} al chunk {chunk_counter + 1}: {e}\n")
            # Gestione errore: pulisci e passa al prossimo dataset
            if os.path.exists(current_temp_model_data_dir):
                shutil.rmtree(current_temp_model_data_dir)
            exec_times.append(round(time.time() - start_time_total, 2))
            labels.append(label + " (Errore)")
            continue  # Passa al prossimo dataset

        # ---- 3) Aggrega i dati spillati e prepara le lookups per modello ----
        # Questa è la fase di Shuffle/Reduce che consolida tutti i dati per modello.
        print("   Fase 2: Aggregazione dati modello da disco...")
        try:
            model_features, prices_lookup, hps_lookup = aggregate_model_data_from_disk(current_temp_model_data_dir)
        except Exception as e:
            print(f"Errore critico durante l'aggregazione dei dati modello da disco per {label}: {e}")
            fout.write(f"Errore critico durante l'aggregazione dei dati modello da disco per {label}: {e}\n")
            if os.path.exists(current_temp_model_data_dir):
                shutil.rmtree(current_temp_model_data_dir)
            exec_times.append(round(time.time() - start_time_total, 2))
            labels.append(label + " (Errore)")
            continue

        # ---- 4) Genera coppie simili O(M^2) ----
        # Questa fase è il collo di bottiglia principale per la RAM se ci sono troppi modelli unici.
        # model_features_dict DEVE stare in RAM.
        print("   Fase 3: Generazione coppie simili (potenziale collo di bottiglia RAM)...")
        similar_pairs_set = generate_similar_pairs(model_features)

        # Trasforma in una struttura per il raggruppamento (simula groupByKey)
        groups_raw = defaultdict(set)
        for m1, m2 in similar_pairs_set:
            groups_raw[m1].add(m2)
            groups_raw[m2].add(m1)  # Assicura collegamento bidirezionale

        # Converti i set in liste ordinate per output consistente
        groups_dict = {model: sorted(list(members)) for model, members in groups_raw.items()}

        # ---- 5) Prepara per i calcoli per gruppo (Explode Groups) ----
        print("   Fase 4: Esplosione gruppi per calcoli finali...")
        exploded_data = list(map_explode_groups(groups_dict))

        # Clear intermediates to save memory where possible
        del groups_raw, similar_pairs_set
        gc.collect()

        # ---- 6) Calcola avg_price per group ----
        print("   Fase 5: Calcolo prezzo medio per gruppo...")
        group_price = reduce_avg_price_by_group(exploded_data, prices_lookup)

        # ---- 7) Calcola top_power_model per group ----
        print("   Fase 6: Calcolo modello più potente per gruppo...")
        group_hp = reduce_top_power_model_by_group(exploded_data, hps_lookup)

        # Clear intermediates to save memory
        del exploded_data, prices_lookup, hps_lookup
        gc.collect()

        # ---- 8) Unisci i risultati e finalizza ----
        print("   Fase 7: Unione risultati finali...")
        final_result = []
        for group_name, members in groups_dict.items():
            avg_price = group_price.get(group_name)
            top_model = group_hp.get(group_name)
            if members and avg_price is not None and top_model:  # Assicurati di avere tutti i componenti
                final_result.append((group_name, members, avg_price, top_model))

        # Ordina il risultato finale per group name per output consistente
        final_result.sort(key=lambda x: x[0])

        duration = round(time.time() - start_time_total, 2)

        # ---- 9) Stampa e log ----
        fout.write("Prime 10 gruppi simili:\n")
        print("Prime 10 gruppi simili:")

        # Stampa solo i primi 10 per brevità
        for i, (grp, members, avgp, topm) in enumerate(final_result):
            if i >= 10:
                break
            line = f"{grp} | members={members} | avg_price={avgp} | top_power_model={topm}"
            print(line)
            fout.write(line + "\n")

        print(f"Tempo ModelSimilarity Locale MapReduce (Spill to Disk): {duration} sec")
        fout.write(f"Tempo ModelSimilarity Locale MapReduce (Spill to Disk): {duration} sec\n\n")

        exec_times.append(duration)
        labels.append(label)

        # Clear all large data structures for the next iteration to free up RAM
        del model_features, group_price, group_hp, groups_dict, final_result
        gc.collect()  # Force garbage collection

        # Pulizia finale della directory temporanea specifica del dataset
        if os.path.exists(current_temp_model_data_dir):
            try:
                shutil.rmtree(current_temp_model_data_dir)
                print(f"   --> Pulita directory temporanea per {label}: {current_temp_model_data_dir}")
            except OSError as e:
                print(f"   --> Errore durante la pulizia finale di {current_temp_model_data_dir}: {e}")

# ---- 10) Grafico dei tempi ----
plt.figure(figsize=(12, 7))
plt.plot(labels, exec_times, marker='o', linestyle='-', color='b')
plt.title("Model Similarity Locale MapReduce: Tempo vs Dimensione Dataset (Spill to Disk)", fontsize=16)
plt.xlabel("Dimensione Dataset (%)", fontsize=12)
plt.ylabel("Tempo di Esecuzione (s)", fontsize=12)
plt.ylim(bottom=0)  # Assicurati che l'asse Y parta da zero
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig(graph_file)
plt.close()

print("\n--- Processo Completato ---")
print(f"Report dettagliato salvato in: {log_file}")
print(f"Grafico dei tempi di esecuzione salvato in: {graph_file}")