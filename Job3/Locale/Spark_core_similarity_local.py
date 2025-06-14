import csv
from io import StringIO
import re
import time
import os
import matplotlib.pyplot as plt
import numpy as np  # Necessario per np.nan

from pyspark import SparkContext
from pyspark.sql import SparkSession  # Mantenuto solo per inizializzare SparkSession

# --- Configurazione per Esecuzione Locale ---
# Le directory locali devono esistere
SPARK_LOCAL_DIR = "/media/gianluigi/Z Slim/Spark_tmp"
os.makedirs(SPARK_LOCAL_DIR, exist_ok=True)

spark_configs = {
    "spark.driver.memory": "3g",
    "spark.executor.memory": "3g",
    "spark.sql.shuffle.partitions": "16",
    "spark.driver.maxResultSize": "1g",
    "spark.default.parallelism": "8",
    "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
    "spark.kryoserializer.buffer.max": "256m",
    "spark.local.dir": SPARK_LOCAL_DIR
}

# Percorsi dei dataset locali
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
    ('/media/gianluigi/Z Slim/DownloadUbuntu/archive/used_cars_data_1_25x.csv', '125%')
]

# ⚡ Inizializza SparkSession con le configurazioni ottimizzate
spark_builder = SparkSession.builder \
    .appName("SimilaritySparkCoreRDDOnlyLocal")

for k, v in spark_configs.items():
    spark_builder = spark_builder.config(k, v)

spark = spark_builder.getOrCreate()

# Imposta il livello di log per avere un output più pulito
spark.sparkContext.setLogLevel("ERROR")
sc = spark.sparkContext

output_dir = "/media/gianluigi/Z Slim/Risultati_locale/"
os.makedirs(output_dir, exist_ok=True)
log_file = os.path.join(output_dir, "model_similarity_core_rdd_only_local.txt")
graph_file = os.path.join(output_dir, "model_similarity_core_rdd_only_local_times.png")

exec_times = []
labels = []

# Definisci le colonne necessarie per il parsing RDD e la loro mappatura
COL_NAMES_MAPPING = {
    'make_name': 'make_name',
    'model_name': 'model_name',
    'horsepower': 'horsepower',
    'engine_displacement': 'engine_displacement',
    'price': 'price',
    'year': 'year'
}


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

        # Mappa i campi a variabili con valori di default o NaN per gestione errori
        model_name = fields[header_map['model_name']].strip() if 'model_name' in header_map and header_map[
            'model_name'] < len(fields) else None

        horsepower = None
        hp_str = fields[header_map['horsepower']].strip() if 'horsepower' in header_map and header_map[
            'horsepower'] < len(fields) else ''
        try:
            # Rimuove simboli valuta, virgole e spazi prima del cast a float
            horsepower = float(re.sub(r'[$,\s]', '', hp_str)) if hp_str else np.nan
        except ValueError:
            horsepower = np.nan

        engine_displacement = None
        ed_str = fields[header_map['engine_displacement']].strip() if 'engine_displacement' in header_map and \
                                                                      header_map['engine_displacement'] < len(
            fields) else ''
        try:
            engine_displacement = float(re.sub(r'[^\d.]', '', ed_str)) if ed_str else np.nan
        except ValueError:
            engine_displacement = np.nan

        price = None
        price_str = fields[header_map['price']].strip() if 'price' in header_map and header_map['price'] < len(
            fields) else ''
        try:
            price = float(re.sub(r'[$,\s]', '', price_str)) if price_str else np.nan
        except ValueError:
            price = np.nan

        year = None
        year_str = fields[header_map['year']].strip() if 'year' in header_map and header_map['year'] < len(
            fields) else ''
        try:
            year = int(float(year_str)) if year_str else -1
        except ValueError:
            year = -1  # Default a -1 per anni non validi

        make_name = fields[header_map['make_name']].strip() if 'make_name' in header_map and header_map[
            'make_name'] < len(fields) else None

        # Applicazione delle condizioni di filtro (identiche a quelle del DataFrame)
        # 1. Colonne essenziali non nulle e valori positivi/validi per numerici
        if not (model_name and model_name.strip() != '' and
                not np.isnan(horsepower) and horsepower > 0 and
                not np.isnan(engine_displacement) and engine_displacement > 0 and
                not np.isnan(price) and price > 0):
            return None

        # 2. Filtri per price e year range
        if not (price > 0 and price < 1_000_000 and
                year >= 1900 and year <= 2025):
            return None

        # 3. Filtro make_name rlike
        if make_name and re.match(r"^[0-9]+(\.[0-9]+)? in$", make_name):
            return None  # Escludi make_name come "123 in"

        return (model_name, horsepower, engine_displacement, price)

    except (csv.Error, IndexError, StopIteration, ValueError) as e:
        # Per debugging, puoi decommentare
        # print(f"Parsing error for line: {line.strip()}. Error: {e}", file=sys.stderr)
        return None
    except Exception as e:
        # Per debugging, puoi decommentare
        # print(f"Unexpected error for line: {line.strip()}. Error: {e}", file=sys.stderr)
        return None


with open(log_file, "w", encoding="utf-8") as fout:
    for path, label in datasets:
        fout.write(f"\n== Dataset {label}: {path} ==\n")
        print(f"\n== Dataset {label}: {path} ==\n")

        try:
            # --- 3) Leggi CSV con Spark RDD ---
            start = time.time()
            raw_rdd = sc.textFile(path)

            if raw_rdd.isEmpty():
                print(f"AVVISO: Il file {path} è vuoto o non accessibile. Salto questo dataset.\n\n")
                fout.write(f"AVVISO: Il file {path} è vuoto o non accessibile. Salto questo dataset.\n\n")
                continue

            header_line = raw_rdd.first()
            data_rdd = raw_rdd.filter(lambda line: line != header_line)

            # Parsa l'header per creare una mappa nome_colonna -> indice
            header_fields = next(csv.reader(StringIO(header_line)))
            header_map = {name.strip(): idx for idx, name in enumerate(header_fields)}

            # Verifica che le colonne essenziali per il parsing siano presenti
            required_cols_for_parse = list(COL_NAMES_MAPPING.keys())  # Usa la mappatura definita
            if not all(col_name in header_map for col_name in required_cols_for_parse):
                missing_cols = [col_name for col_name in required_cols_for_parse if col_name not in header_map]
                print(
                    f"ERRORE: Colonne essenziali mancanti nell'header di {path}: {missing_cols}. Salto questo dataset.\n\n")
                fout.write(
                    f"ERRORE: Colonne essenziali mancanti nell'header di {path}: {missing_cols}. Salto questo dataset.\n\n")
                continue

        except Exception as e:
            print(
                f"ERRORE: Impossibile leggere o processare l'header del file {path}. Salto questo dataset. Errore: {e}\n\n")
            fout.write(
                f"ERRORE: Impossibile leggere o processare l'header del file {path}. Salto questo dataset. Errore: {e}\n\n")
            continue

        header_map_broadcast = sc.broadcast(header_map)

        # ---- 4) Pulizia e filtra: RDD[(model_name, horsepower, engine_displacement, price)] ----
        rdd_filtered_data = data_rdd.map(lambda line: parse_and_filter_csv_line(line, header_map_broadcast.value)) \
            .filter(lambda x: x is not None) \
            .cache()

        if rdd_filtered_data.isEmpty():
            print(f"AVVISO: Nessun dato valido dopo pulizia e filtro per {path}. Salto questo dataset.\n\n")
            fout.write(f"AVVISO: Nessun dato valido dopo pulizia e filtro per {path}. Salto questo dataset.\n\n")
            header_map_broadcast.destroy()
            continue

        # ---- 5) Converti in RDD[(model, (hp, ed, price))] ----
        rdd_model_raw_data = rdd_filtered_data.map(lambda r: (
            r[0],  # model_name
            (r[1], r[2], r[3])  # (horsepower, engine_displacement, price)
        )).cache()

        # ---- 6) Calcola model_feat: [(model, (avg_hp, avg_ed))] ----
        # Questo RDD viene collectato al driver. Punto critico di scalabilità.
        mf_collected_list = (rdd_model_raw_data
                             .map(lambda x: (x[0], (x[1][0], x[1][1], 1)))  # (model, (sum_hp, sum_ed, count))
                             .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1], a[2] + b[2]))
                             .mapValues(
            lambda t: (t[0] / t[2], t[1] / t[2]) if t[2] > 0 else (0.0, 0.0))  # (avg_hp, avg_ed)
                             .collect()  # <<< CRITICAL POINT: Questo porta i dati al driver.
                             )

        bc_feats = sc.broadcast(mf_collected_list)


        # ---- 7) Genera coppie simili O(M^2) ----
        # Questa trasformazione è eseguita sul driver.
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


        similar_pairs_rdd = sc.parallelize([0], numPartitions=1).flatMap(gen_pairs_driver_side).distinct().cache()

        if similar_pairs_rdd.isEmpty():
            print(f"AVVISO: Nessuna coppia di modelli simili trovata per {path}. Salto questo dataset.\n\n")
            fout.write(f"AVVISO: Nessuna coppia di modelli simili trovata per {path}. Salto questo dataset.\n\n")
            rdd_filtered_data.unpersist()
            rdd_model_raw_data.unpersist()
            bc_feats.destroy()
            header_map_broadcast.destroy()
            continue

        # ---- 8) Raggruppa membri per each group_model (Connected Components) ----
        groups_rdd = similar_pairs_rdd.groupByKey() \
            .mapValues(lambda it: list(sorted(set(it)))) \
            .cache()

        # ---- 9) Prepara RDD per price e hp lookup ----
        prices_rdd = rdd_model_raw_data.map(lambda x: (x[0], x[1][2])).cache()
        hps_rdd = rdd_model_raw_data.map(lambda x: (x[0], x[1][0])).cache()

        # ---- 10) Esplode: (member, group_representative) per join ----
        exploded_memberships_rdd = groups_rdd.flatMap(lambda x: [(member, x[0]) for member in x[1]])

        # ---- 11) Calcola avg_price per group ----
        group_price_rdd = (exploded_memberships_rdd.join(prices_rdd)  # (member, (group, price))
                           .map(lambda x: (x[1][0], (x[1][1], 1)))  # key=group, value=(price, 1)
                           .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
                           .mapValues(lambda t: round(t[0] / t[1], 2) if t[1] > 0 else 0.0)
                           .cache()
                           )

        # ---- 12) Calcola top_power_model per group ----
        group_hp_rdd = (exploded_memberships_rdd.join(hps_rdd)  # (member, (group, hp))
                        .map(lambda x: (x[1][0], (x[0], x[1][1])))  # (group, (member, hp))
                        .groupByKey()
                        .mapValues(lambda seq: max(seq, key=lambda y: y[1])[0])
                        .cache()
                        )

        # ---- 13) Unisci e misura tempi ----
        result_rdd = (group_price_rdd
                      .join(groups_rdd)  # (group, (avg_price, members_list))
                      .join(group_hp_rdd)  # (group, ((avg_price, members_list), top_model))
                      .map(lambda x: (
            x[0],  # group (key del gruppo)
            x[1][0][1],  # members (lista dei membri del gruppo)
            x[1][0][0],  # avg_price (prezzo medio del gruppo)
            x[1][1]  # top_power_model (modello con più HP nel gruppo)
        ))
                      .sortBy(lambda x: x[0])
                      )
        sample = result_rdd.take(10)  # take() forza l'esecuzione
        duration = round(time.time() - start, 2)

        # ---- 14) Stampa e log ----
        fout.write("Prime 10 gruppi simili:\n")
        print("Prime 10 gruppi simili:")
        for grp, members, avgp, topm in sample:
            line = f"Group: {grp} | Members: {members} | Avg Price: {avgp} | Top Power Model: {topm}"
            print(line)
            fout.write(line + "\n")

        print(f"Tempo ModelSimilarity Spark Core (RDD Only, Local): {duration} sec")
        fout.write(f"Tempo ModelSimilarity Spark Core (RDD Only, Local): {duration} sec\n\n")

        exec_times.append(duration)
        labels.append(label)

        # Rilascia la cache dei RDD e broadcast variables
        rdd_filtered_data.unpersist()
        rdd_model_raw_data.unpersist()
        similar_pairs_rdd.unpersist()
        groups_rdd.unpersist()
        prices_rdd.unpersist()
        hps_rdd.unpersist()
        group_price_rdd.unpersist()
        group_hp_rdd.unpersist()
        bc_feats.destroy()
        header_map_broadcast.destroy()

    # ---- 15) Grafico dei tempi ----
plt.figure(figsize=(8, 5))
plt.plot(labels, exec_times, marker='o', linestyle='-')
plt.title("Model Similarity Spark Core (RDD Only, Local): Tempo vs Dataset Size")
plt.xlabel("Sample size")
plt.ylabel("Execution time (s)")
plt.ylim(0, max(exec_times) + 10)
plt.grid(True)
plt.tight_layout()
plt.savefig(graph_file)
plt.close()

print(f"\n✅ Report e grafico salvati in: {output_dir}")

# Termina la sessione Spark
spark.stop()