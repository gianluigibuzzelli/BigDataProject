from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_replace, col, when, isnan
import time, os, csv
import sys  # Importa sys per output su stderr/stdout del driver
import matplotlib.pyplot as plt
import numpy as np

# --- Configurazione per Cluster EMR ---
# Su EMR, i percorsi devono essere S3, non locali.
S3_INPUT_BASE_PATH = "s3://bucketpoggers2/input/"

# Il file di log e il file dei tempi verranno generati localmente sul nodo master EMR
# e poi scaricati.
LOCAL_LOG_DIR = "/home/hadoop/spark_logs_model_similarity/"  # Directory sul nodo master EMR per i log temporanei
LOCAL_LOG_FILE = os.path.join(LOCAL_LOG_DIR, "model_similarity_core.txt")
LOCAL_TIMES_FILE_ON_EMR = os.path.join(LOCAL_LOG_DIR, "model_similarity_core_times.txt")

# Assicurati che la directory di log esista sul nodo driver EMR
os.makedirs(LOCAL_LOG_DIR, exist_ok=True)

# ⚡ Inizializza SparkSession. Senza configurazioni esplicite.
# Spark userà le configurazioni di default del cluster EMR.
spark = SparkSession.builder \
    .appName("ModelSimilaritySparkCluster") \
    .getOrCreate()

# Imposta il livello di log per avere un output più pulito
spark.sparkContext.setLogLevel("ERROR")
sc = spark.sparkContext

# Definizione dei dataset (nomi file S3 e etichette per il grafico)
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

with open(LOCAL_LOG_FILE, "w", encoding="utf-8") as fout:
    for file_name, label in datasets:
        s3_path = S3_INPUT_BASE_PATH + file_name
        fout.write(f"\n== Dataset {label}: {s3_path} ==\n")
        sys.stdout.write(f"\n== Dataset {label}: {s3_path} ==\n")

        # ---- 3) Leggi CSV con Spark DataFrame ----
        # Spark legge direttamente da S3.
        try:
            df = spark.read.option("header", True) \
                .option("inferSchema", True) \
                .option("multiLine", True) \
                .option("quote", "\"") \
                .option("escape", "\\") \
                .csv(s3_path)

            # Verifica se il DataFrame è vuoto dopo la lettura
            if df.head(1) == []:  # Usare df.count() può essere costoso
                sys.stdout.write(f"AVVISO: Il file {s3_path} è vuoto o non accessibile. Salto questo dataset.\n\n")
                fout.write(f"AVVISO: Il file {s3_path} è vuoto o non accessibile. Salto questo dataset.\n\n")
                continue

        except Exception as e:
            sys.stdout.write(
                f"ERRORE: Impossibile leggere o processare il file {s3_path}. Salto questo dataset. Errore: {e}\n\n")
            fout.write(
                f"ERRORE: Impossibile leggere o processare il file {s3_path}. Salto questo dataset. Errore: {e}\n\n")
            continue

        # ---- 4) Pulisci e filtra ----
        # Utilizzo isnan per double/float e isNull per tutte le colonne
        df_clean = df \
            .withColumn("price", regexp_replace(col("price"), "[$,\\s]", "").cast("double")) \
            .withColumn("horsepower", col("horsepower").cast("double")) \
            .withColumn("engine_displacement", col("engine_displacement").cast("double")) \
            .filter(col("model_name").isNotNull()) \
            .filter(~isnan(col("horsepower")) & col("horsepower").isNotNull() & (col("horsepower") > 0)) \
            .filter(~isnan(col("engine_displacement")) & col("engine_displacement").isNotNull() & (
                    col("engine_displacement") > 0)) \
            .filter(~isnan(col("price")) & col("price").isNotNull() & (col("price") > 0)) \
            .cache()  # Cache il DataFrame pulito per riutilizzo

        if df_clean.head(1) == []:
            sys.stdout.write(
                f"AVVISO: Nessun dato valido dopo pulizia e filtro per {s3_path}. Salto questo dataset.\n\n")
            fout.write(f"AVVISO: Nessun dato valido dopo pulizia e filtro per {s3_path}. Salto questo dataset.\n\n")
            df_clean.unpersist()
            continue

        # ---- 5) Converti in RDD[(model, (hp, ed, price))] ----
        rdd = df_clean.select("model_name", "horsepower", "engine_displacement", "price").rdd.map(lambda r: (
            r.model_name,
            (r.horsepower, r.engine_displacement, r.price)
        ))

        # ---- 6) Calcola model_feat: RDD[(model, (avg_hp, avg_ed))] ----
        mf = (rdd
              .map(lambda x: (x[0], (x[1][0], x[1][1], 1)))  # (model, (sum_hp, sum_ed, count))
              .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1], a[2] + b[2]))
              .mapValues(lambda t: (t[0] / t[2], t[1] / t[2]))  # (avg_hp, avg_ed)
              .collect()  # Collect to driver: attenzioni a dataset molto grandi
              )
        # La `collect()` qui è un punto critico di scalabilità. Se il numero di modelli unici
        # `M` è molto grande, `mf` potrebbe non stare in memoria sul driver.
        # Per dataset molto grandi, questa parte dovrebbe essere gestita diversamente,
        # ad esempio usando un join cartesian o GraphFrames.

        bc_feats = sc.broadcast(mf)


        # ---- 7) Genera coppie simili O(M^2) ----
        # Questa trasformazione è eseguita sul driver.
        # `sc.parallelize([0])` è un modo per innescare un task su un executor
        # per accedere alla broadcast variable, ma la logica O(M^2) si manifesta sul driver.
        # Se `M` è grande, `gen_pairs` sarà molto lento e intensivo in memoria sul driver.
        # Una soluzione più scalabile sarebbe generare queste coppie direttamente su RDD distribuiti,
        # o usare GraphFrames per la ricerca di community/componenti connesse basate sulla similarità.
        def gen_pairs(_):
            out = []
            feats = bc_feats.value  # Accesso ai modelli e loro feature

            # Loop nidificati per O(M^2)
            for i in range(len(feats)):
                m1, (hp1, ed1) = feats[i]
                for j in range(i + 1, len(feats)):  # i+1 per evitare duplicati e auto-connessioni
                    m2, (hp2, ed2) = feats[j]

                    # Calcola la similarità, gestendo divisione per zero se hp1 o ed1 sono zero (se non già filtrati)
                    hp_diff_ratio = abs(hp1 - hp2) / hp1 if hp1 != 0 else float('inf')
                    ed_diff_ratio = abs(ed1 - ed2) / ed1 if ed1 != 0 else float('inf')

                    if hp_diff_ratio <= 0.10 and ed_diff_ratio <= 0.10:
                        out.append((m1, m2))
                        out.append((m2, m1))  # Aggiungi anche la coppia inversa per groupBy
            return out


        similar = sc.parallelize([0], numPartitions=1).flatMap(
            gen_pairs).distinct()  # `numPartitions=1` per assicurarsi che gen_pairs sia eseguito una volta
        # `distinct()` per rimuovere duplicati se generati

        # ---- 8) Raggruppa membri per each group_model (Connected Components) ----
        # Qui il groupBy e mapValues crea i gruppi di modelli simili.
        # Questo è il cuore della "Connected Components" per i modelli.
        groups = similar.groupByKey() \
            .mapValues(lambda it: list(sorted(set(it)))) \
            .cache()  # Cache il risultato dei gruppi

        if groups.isEmpty():
            sys.stdout.write(
                f"AVVISO: Nessun gruppo di modelli simili trovato per {s3_path}. Salto questo dataset.\n\n")
            fout.write(f"AVVISO: Nessun gruppo di modelli simili trovato per {s3_path}. Salto questo dataset.\n\n")
            df_clean.unpersist()
            bc_feats.destroy()
            continue

        # ---- 9) Prepara RDD per price e hp lookup ----
        # RDD[(model, price)] e RDD[(model, hp)] per join successivi
        prices = rdd.map(lambda x: (x[0], x[1][2])).cache()
        hps = rdd.map(lambda x: (x[0], x[1][0])).cache()

        # ---- 10) Esplode: (member, group) per join ----
        # Trasforma i gruppi in una lista di (modello, modello_rappresentante_del_gruppo)
        # E.g., se il gruppo è (A, [A,B,C]), crea (A,A), (B,A), (C,A)
        exploded = groups.flatMap(lambda x: [(member, x[0]) for member in x[1]])  # x[0] è il key del gruppo

        # ---- 11) Calcola avg_price per group ----
        group_price = (exploded.join(prices)  # (member, (group, price)) -> (member, (group, price))
                       .map(lambda x: (x[1][0], (x[1][1], 1)))  # key=group, value=(price, 1)
                       .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
                       .mapValues(
            lambda t: round(t[0] / t[1], 2) if t[1] > 0 else 0.0)  # Calcola media, gestisci divisione per zero
                       .cache()
                       )  # RDD[(group, group_avg_price)]

        # ---- 12) Calcola top_power_model per group ----
        group_hp = (exploded.join(hps)  # (member, (group, hp)) -> (member, (group, hp))
                    .map(lambda x: (x[1][0], (x[0], x[1][1])))  # key=group, value=(member, hp)
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
                  .sortBy(lambda x: x[0])  # Ordina per il nome del gruppo
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

        sys.stdout.write(f"Tempo Model Similarity Spark Core: {duration} sec\n\n")
        fout.write(f"Tempo Model Similarity Spark Core: {duration} sec\n\n")

        exec_times.append(duration)
        labels.append(label)

        # Rilascia la cache dei RDD/DataFrame e broadcast variables
        df_clean.unpersist()
        # mf (collectato) non è un RDD da unpersist()
        bc_feats.destroy()  # Importante liberare la memoria broadcast
        groups.unpersist()
        prices.unpersist()
        hps.unpersist()
        group_price.unpersist()
        group_hp.unpersist()

# Scrivi i tempi di esecuzione in un file separato sul nodo master EMR
with open(LOCAL_TIMES_FILE_ON_EMR, "w", encoding="utf-8") as tf:
    for i in range(len(labels)):
        tf.write(f"{labels[i]} {exec_times[i]}\n")

# ---- 15) Arresta SparkSession ----
spark.stop()

sys.stdout.write(f"\n✅ Report salvato in: {LOCAL_LOG_FILE}\n")
sys.stdout.write(f"✅ Tempi di esecuzione salvati in: {LOCAL_TIMES_FILE_ON_EMR}\n")
sys.stdout.write(f"Per generare il grafico, scarica '{LOCAL_TIMES_FILE_ON_EMR}' e usa uno script Python locale.\n")