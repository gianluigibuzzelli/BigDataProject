import csv
from io import StringIO
import re
import time
import os
import sys  # Importa sys per output su stderr/stdout del driver
import matplotlib.pyplot as plt
import numpy as np

from pyspark import SparkContext
from pyspark.sql import SparkSession  # Mantenuto per inizializzare SparkSession

# --- Configurazione per Cluster EMR ---
# Su EMR, i percorsi devono essere S3, non locali.
S3_CLEANED_INPUT_BASE_PATH = "s3://bucketpoggers2/input/"

# Il file di log e il file dei tempi verranno generati localmente sul nodo master EMR
# e poi scaricati.
LOCAL_LOG_DIR = "/home/hadoop/spark_logs_model_similarity_rdd_only/"
LOCAL_LOG_FILE = os.path.join(LOCAL_LOG_DIR, "model_similarity_core_rdd_only.txt")
LOCAL_TIMES_FILE_ON_EMR = os.path.join(LOCAL_LOG_DIR, "model_similarity_core_rdd_only_times.txt")

# Assicurati che la directory di log esista sul nodo driver EMR
os.makedirs(LOCAL_LOG_DIR, exist_ok=True)

# ⚡ Inizializza SparkSession e ottieni SparkContext.
spark = SparkSession.builder \
    .appName("ModelSimilarityRDDCluster") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
sc = spark.sparkContext

# La lista dei dataset è stata aggiornata per puntare ai file puliti nella directory 'cleaned'
datasets = [
    ('used_cars_data_sampled_1.csv', '10%'),
    ('used_cars_data_sampled_2.csv', '20%'),
    ('used_cars_data_sampled_3.csv', '30%'),
    ('used_cars_data_sampled_4.csv', '40%'),
    ('used_cars_data_sampled_5.csv', '50%'),
    ('used_cars_data_sampled_6.csv', '60%'),
    ('used_cars_data_sampled_7.csv', '70%'),
    ('used_cars_data_sampled_8.csv', '80%'),
    ('used_cars_data_sampled_9.csv', '90%'),
    ('used_cars_data.csv', '100%'),
    ('used_cars_data_1_25x.csv', '125%')
]

exec_times = []
labels = []


def parse_and_filter_csv_line(line: str, header_map: dict):
    """
    Parsa una singola riga CSV e applica i filtri e le conversioni di tipo.
    Args:
        line (str): La riga CSV da parsare.
        header_map (dict): Una mappa da nome colonna a indice.
    Returns:
        tuple: Una tupla nel formato (model_name, horsepower, engine_displacement, price)
               o None se la riga è malformata o non supera i filtri.
    """
    try:
        reader = csv.reader(StringIO(line))
        fields = next(reader)

        # Verifica che la riga abbia un numero sufficiente di campi
        required_indices = [
            header_map.get('model_name', -1),
            header_map.get('horsepower', -1),
            header_map.get('engine_displacement', -1),
            header_map.get('price', -1)
        ]
        if any(idx == -1 for idx in required_indices) or len(fields) <= max(required_indices):
            return None  # Colonne richieste mancanti nell'header o riga troppo corta

        model_name = fields[header_map['model_name']].strip() if 'model_name' in header_map else None

        horsepower = None
        hp_str = fields[header_map['horsepower']].strip() if 'horsepower' in header_map else ''
        try:
            # Assumendo che i file puliti non abbiano simboli valuta, solo numeri o stringa vuota
            horsepower = float(hp_str) if hp_str else np.nan
        except ValueError:
            horsepower = np.nan

        engine_displacement = None
        ed_str = fields[header_map['engine_displacement']].strip() if 'engine_displacement' in header_map else ''
        try:
            engine_displacement = float(ed_str) if ed_str else np.nan
        except ValueError:
            engine_displacement = np.nan

        price = None
        price_str = fields[header_map['price']].strip() if 'price' in header_map else ''
        try:
            price = float(price_str) if price_str else np.nan
        except ValueError:
            price = np.nan

        # Applicazione delle condizioni di filtro (simili a quelle del DataFrame)
        # isnan gestisce np.nan, is not None per i casi non numerici (se il cast fallisce in modi inaspettati)
        if not (model_name and model_name.strip() != '' and
                not np.isnan(horsepower) and horsepower > 0 and
                not np.isnan(engine_displacement) and engine_displacement > 0 and
                not np.isnan(price) and price > 0):
            return None

        return (model_name, horsepower, engine_displacement, price)

    except (csv.Error, IndexError, StopIteration, ValueError) as e:
        # Cattura errori di parsing CSV, accesso a indici fuori bounds, righe vuote o conversioni
        # sys.stderr.write(f"Parsing error for line: {line.strip()}. Error: {e}\n") # Debugging
        return None
    except Exception as e:
        # Cattura qualsiasi altro errore imprevisto
        # sys.stderr.write(f"Unexpected error for line: {line.strip()}. Error: {e}\n") # Debugging
        return None


with open(LOCAL_LOG_FILE, "w", encoding="utf-8") as fout:
    for file_name, label in datasets:
        s3_path = S3_CLEANED_INPUT_BASE_PATH + file_name
        fout.write(f"\n== Dataset {label}: {s3_path} ==\n")
        sys.stdout.write(f"\n== Dataset {label}: {s3_path} ==\n")

        try:
            # --- 3) Leggi CSV con Spark RDD ---
            raw_rdd = sc.textFile(s3_path)

            if raw_rdd.isEmpty():
                sys.stdout.write(f"AVVISO: Il file {s3_path} è vuoto o non accessibile. Salto questo dataset.\n\n")
                fout.write(f"AVVISO: Il file {s3_path} è vuoto o non accessibile. Salto questo dataset.\n\n")
                continue

            header_line = raw_rdd.first()
            data_rdd = raw_rdd.filter(lambda line: line != header_line)

            # Parsa l'header per creare una mappa nome_colonna -> indice
            header_fields = next(csv.reader(StringIO(header_line)))
            header_map = {name.strip(): idx for idx, name in enumerate(header_fields)}

            required_cols = ['model_name', 'horsepower', 'engine_displacement', 'price']
            if not all(col_name in header_map for col_name in required_cols):
                missing_cols = [col_name for col_name in required_cols if col_name not in header_map]
                sys.stdout.write(
                    f"ERRORE: Colonne essenziali mancanti nell'header di {s3_path}: {missing_cols}. Salto questo dataset.\n\n")
                fout.write(
                    f"ERRORE: Colonne essenziali mancanti nell'header di {s3_path}: {missing_cols}. Salto questo dataset.\n\n")
                continue

        except Exception as e:
            sys.stdout.write(
                f"ERRORE: Impossibile leggere o processare l'header del file {s3_path}. Salto questo dataset. Errore: {e}\n\n")
            fout.write(
                f"ERRORE: Impossibile leggere o processare l'header del file {s3_path}. Salto questo dataset. Errore: {e}\n\n")
            continue

        # Broadcast di header_map per renderlo disponibile efficientemente su tutti i worker
        header_map_broadcast = sc.broadcast(header_map)

        # ---- 4) Applica parsing e filtraggio (sostituisce df_clean) ----
        # rdd_filtered: RDD[(model_name, horsepower, engine_displacement, price)]
        rdd_filtered = data_rdd.map(lambda line: parse_and_filter_csv_line(line, header_map_broadcast.value)) \
            .filter(lambda x: x is not None) \
            .cache()  # Cache il RDD filtrato

        if rdd_filtered.isEmpty():
            sys.stdout.write(
                f"AVVISO: Nessun dato valido dopo parsing e filtro per {s3_path}. Salto questo dataset.\n\n")
            fout.write(f"AVVISO: Nessun dato valido dopo parsing e filtro per {s3_path}. Salto questo dataset.\n\n")
            header_map_broadcast.destroy()
            continue

        # ---- 5) Converti in RDD[(model, (hp, ed, price))] ----
        # Questo RDD è già in questo formato quasi dal parsing
        # (model_name, horsepower, engine_displacement, price)
        # Dobbiamo solo mappare leggermente per ottenere la tupla (hp, ed, price) come valore
        rdd_model_data = rdd_filtered.map(lambda r: (
            r[0],  # model_name
            (r[1], r[2], r[3])  # (horsepower, engine_displacement, price)
        )).cache()  # Cache anche questo RDD se riutilizzato

        # ---- 6) Calcola model_feat: [(model, (avg_hp, avg_ed))] ----
        mf_collect = (rdd_model_data
                      .map(lambda x: (x[0], (x[1][0], x[1][1], 1)))  # (model, (sum_hp, sum_ed, count))
                      .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1], a[2] + b[2]))
                      .mapValues(lambda t: (t[0] / t[2], t[1] / t[2]) if t[2] > 0 else (0.0, 0.0))  # (avg_hp, avg_ed)
                      .collect()  # <<< CRITICAL POINT: Questo porta i dati al driver. Scalabilità limitata.
                      )

        bc_feats = sc.broadcast(mf_collect)


        # ---- 7) Genera coppie simili O(M^2) ----
        # Questa logica è eseguita sul driver.
        def gen_pairs_driver_side(_):
            out = []
            feats = bc_feats.value

            for i in range(len(feats)):
                m1, (hp1, ed1) = feats[i]
                for j in range(i + 1, len(feats)):
                    m2, (hp2, ed2) = feats[j]

                    hp_diff_ratio = abs(hp1 - hp2) / hp1 if hp1 != 0 else float('inf')
                    ed_diff_ratio = abs(ed1 - ed2) / ed1 if ed1 != 0 else float('inf')

                    if hp_diff_ratio <= 0.10 and ed_diff_ratio <= 0.10:
                        out.append((m1, m2))
                        out.append((m2, m1))  # Aggiungi anche la coppia inversa per groupBy
            return out


        # `numPartitions=1` assicura che `gen_pairs_driver_side` sia eseguito una sola volta sul driver.
        # `distinct()` per rimuovere eventuali duplicati di coppie.
        similar_pairs_rdd = sc.parallelize([0], numPartitions=1).flatMap(gen_pairs_driver_side).distinct().cache()

        if similar_pairs_rdd.isEmpty():
            sys.stdout.write(
                f"AVVISO: Nessuna coppia di modelli simili trovata per {s3_path}. Salto questo dataset.\n\n")
            fout.write(f"AVVISO: Nessuna coppia di modelli simili trovata per {s3_path}. Salto questo dataset.\n\n")
            rdd_filtered.unpersist()
            rdd_model_data.unpersist()
            bc_feats.destroy()
            continue

        # ---- 8) Raggruppa membri per each group_model (Connected Components) ----
        # groups: RDD[(group_representative_model, list_of_members_in_group)]
        groups = similar_pairs_rdd.groupByKey() \
            .mapValues(lambda it: list(sorted(set(it)))) \
            .cache()

        # ---- 9) Prepara RDD per price e hp lookup ----
        # prices: RDD[(model, price)]
        # hps: RDD[(model, hp)]
        # Questi provengono direttamente da rdd_model_data.
        prices = rdd_model_data.map(lambda x: (x[0], x[1][2])).cache()
        hps = rdd_model_data.map(lambda x: (x[0], x[1][0])).cache()

        # ---- 10) Esplode: (member, group_representative) per join ----
        exploded_memberships = groups.flatMap(lambda x: [(member, x[0]) for member in x[1]])

        # ---- 11) Calcola avg_price per group ----
        group_price = (exploded_memberships.join(prices)  # (member, (group, price))
                       .map(lambda x: (x[1][0], (x[1][1], 1)))  # key=group, value=(price, 1)
                       .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
                       .mapValues(lambda t: round(t[0] / t[1], 2) if t[1] > 0 else 0.0)
                       .cache()
                       )  # RDD[(group, group_avg_price)]

        # ---- 12) Calcola top_power_model per group ----
        group_hp = (exploded_memberships.join(hps)  # (member, (group, hp))
                    .map(lambda x: (x[1][0], (x[0], x[1][1])))  # (group, (member, hp))
                    .groupByKey()
                    .mapValues(lambda seq: max(seq, key=lambda y: y[1])[0])  # Trova il modello con max hp nel gruppo
                    .cache()
                    )  # RDD[(group, top_power_model)]

        # ---- 13) Unisci e misura tempi ----
        start = time.time()
        result = (group_price
                  .join(groups)  # (group, (avg_price, members_list))
                  .join(group_hp)  # (group, ((avg_price, members_list), top_model))
                  .map(lambda x: (
            x[0],  # group (key del gruppo)
            x[1][0][1],  # members (lista dei membri del gruppo)
            x[1][0][0],  # avg_price (prezzo medio del gruppo)
            x[1][1]  # top_power_model (modello con più HP nel gruppo)
        ))
                  .sortBy(lambda x: x[0])
                  )
        sample = result.take(10)  # take() forza l'esecuzione
        duration = round(time.time() - start, 2)

        # ---- 14) Stampa e log ----
        fout.write("Prime 10 gruppi simili:\n")
        sys.stdout.write("Prime 10 gruppi simili:\n")
        for grp, members, avgp, topm in sample:
            line = f"Group: {grp} | Members: {members} | Avg Price: {avgp} | Top Power Model: {topm}"
            sys.stdout.write(line + "\n")
            fout.write(line + "\n")

        sys.stdout.write(f"Tempo Model Similarity Spark Core (RDD Only): {duration} sec\n\n")
        fout.write(f"Tempo Model Similarity Spark Core (RDD Only): {duration} sec\n\n")

        exec_times.append(duration)
        labels.append(label)

        # Rilascia la cache dei RDD e broadcast variables
        rdd_filtered.unpersist()
        rdd_model_data.unpersist()
        similar_pairs_rdd.unpersist()
        groups.unpersist()
        prices.unpersist()
        hps.unpersist()
        group_price.unpersist()
        group_hp.unpersist()
        bc_feats.destroy()  # Importante liberare la memoria broadcast
        header_map_broadcast.destroy()  # Anche la mappa header broadcast

# Scrivi i tempi di esecuzione in un file separato sul nodo master EMR
with open(LOCAL_TIMES_FILE_ON_EMR, "w", encoding="utf-8") as tf:
    for i in range(len(labels)):
        tf.write(f"{labels[i]} {exec_times[i]}\n")

# ---- 15) Arresta SparkSession ----
spark.stop()

sys.stdout.write(f"\n✅ Report salvato in: {LOCAL_LOG_FILE}\n")
sys.stdout.write(f"✅ Tempi di esecuzione salvati in: {LOCAL_TIMES_FILE_ON_EMR}\n")
sys.stdout.write(f"Per generare il grafico, scarica '{LOCAL_TIMES_FILE_ON_EMR}' e usa uno script Python locale.\n")