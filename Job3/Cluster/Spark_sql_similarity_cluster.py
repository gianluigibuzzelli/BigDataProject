from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, regexp_replace, isnan, abs, array_sort
import time
import os
import sys  # Importa sys per output su stderr/stdout del driver
import matplotlib.pyplot as plt  # Mantenuto per la logica di generazione del grafico, ma il plot verrà fatto localmente
import numpy as np  # Mantenuto per la logica di generazione del grafico

# --- Configurazione per Cluster EMR ---
# Su EMR, i percorsi devono essere S3, non locali.
S3_INPUT_BASE_PATH = "s3://bucketpoggers2/input/"  # Assicurati che i tuoi file siano qui

# Il file di log e il file dei tempi verranno generati localmente sul nodo master EMR
LOCAL_LOG_DIR = "/home/hadoop/spark_logs_model_similarity/"  # Directory sul nodo master EMR per i log temporanei
LOCAL_LOG_FILE = os.path.join(LOCAL_LOG_DIR, "model_similarity_results.txt")
LOCAL_TIMES_FILE_ON_EMR = os.path.join(LOCAL_LOG_DIR, "model_similarity_exec_times.txt")

# Assicurati che la directory di log esista sul nodo driver EMR
os.makedirs(LOCAL_LOG_DIR, exist_ok=True)

# I percorsi sono ora relativi alla S3_INPUT_BASE_PATH
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
    ('used_cars_data_2x.csv', '200%')
]

#Inizializza SparkSession. Senza configurazioni esplicite.
# Spark userà le configurazioni di default del cluster EMR.
spark = SparkSession.builder \
    .appName("ModelSimilarityEngineCluster") \
    .getOrCreate()

# Imposta il livello di log per avere un output più pulito
spark.sparkContext.setLogLevel("ERROR")

essential = ['make_name', 'model_name', 'horsepower', 'engine_displacement']
int_cols = ['daysonmarket', 'dealer_zip', 'listing_id', 'owner_count', 'maximum_seating', 'year']

exec_times = []
labels = []

# 📁 Prepara directory e file di output (su nodo master EMR)
with open(LOCAL_LOG_FILE, "w", encoding="utf-8") as f_out:
    for file_name, label in datasets:
        s3_path = S3_INPUT_BASE_PATH + file_name
        f_out.write(f"\n== Dataset {label}: {s3_path} ==\n")
        sys.stdout.write(f"\n== Dataset {label}: {s3_path} ==\n")

        try:
            df = spark.read.csv(s3_path, header=True, inferSchema=True)

            if df.head(1) == []:  # Verifica se il DataFrame è vuoto dopo la lettura
                sys.stdout.write(f"AVVISO: Il file {s3_path} è vuoto o non accessibile. Salto questo dataset.\n\n")
                f_out.write(f"AVVISO: Il file {s3_path} è vuoto o non accessibile. Salto questo dataset.\n\n")
                continue

        except Exception as e:
            sys.stdout.write(
                f"ERRORE: Impossibile leggere o processare il file {s3_path}. Salto questo dataset. Errore: {e}\n\n")
            f_out.write(
                f"ERRORE: Impossibile leggere o processare il file {s3_path}. Salto questo dataset. Errore: {e}\n\n")
            continue

        # Pulizia base e casting dei tipi (identica al tuo script)
        df = df.withColumn("price", regexp_replace("price", "[$,\\s]", "").cast("double"))

        # Filtro più robusto per valori nulli/NaN per le colonne essenziali
        # e per le colonne numeriche critiche (horsepower, engine_displacement, price)
        df_clean = df.filter(
            col("make_name").isNotNull() & col("model_name").isNotNull() &
            ~isnan(col("horsepower")) & col("horsepower").isNotNull() & (col("horsepower") > 0) &
            ~isnan(col("engine_displacement")) & col("engine_displacement").isNotNull() & (
                        col("engine_displacement") > 0)
        )

        # Gestione valori nulli/NaN per colonne intere, sostituendo con -1
        for c in int_cols:
            df_clean = df_clean.withColumn(
                c,
                when(col(c).isNull() | isnan(col(c)) | ~col(c).cast("string").rlike("^-?\\d+$"), -1).otherwise(
                    col(c).cast("int"))
            )

        df_clean = df_clean.withColumn(
            "description",
            when(col("description").isNull(), "").otherwise(col("description"))
        ).filter(
            (col("price") > 0) &
            (col("price") < 1_000_000) &
            (col("year") >= 1900) & (col("year") <= 2025)
        ).filter(~col("make_name").rlike("^[0-9]+(\\.[0-9]+)? in$"))

        # 🚀 Ottimizzazione: Caching del DataFrame pulito e filtrato
        df_clean.cache()
        # Forza il calcolo per popolare la cache. Utile in fase di benchmark.
        df_clean.count()

        if df_clean.head(1) == []:
            sys.stdout.write(
                f"AVVISO: Nessun dato valido dopo pulizia e filtro per {s3_path}. Salto questo dataset.\n\n")
            f_out.write(f"AVVISO: Nessun dato valido dopo pulizia e filtro per {s3_path}. Salto questo dataset.\n\n")
            df_clean.unpersist()
            continue

        df_clean.createOrReplaceTempView("cars")

        # --- similarity job in Spark SQL ---
        # La query è stata leggermente modificata per usare array_sort su group_members
        # per una rappresentazione consistente in output.
        sql = """
        WITH model_feat AS (
          SELECT model_name,
                AVG(horsepower) AS avg_hp,
                AVG(engine_displacement) AS avg_ed
          FROM cars
          GROUP BY model_name
        ),
        similar_pairs AS (
          SELECT a.model_name AS m1, b.model_name AS m2
          FROM model_feat a
          JOIN model_feat b
            ON a.model_name <= b.model_name  -- Per evitare duplicati (m1,m2) e (m2,m1)
          AND ABS(a.avg_hp - b.avg_hp)/a.avg_hp <= 0.10
          AND ABS(a.avg_ed - b.avg_ed)/a.avg_ed <= 0.10
        ),
        -- Per gestire correttamente i gruppi connessi (simili A-B, B-C => A-B-C)
        -- Questa parte è complessa in SQL puro e di solito si farebbe con GraphFrames
        -- o un algoritmo iterativo. Qui, stiamo solo prendendo le coppie dirette e raggruppando.
        -- Per un vero algoritmo di componenti connesse, questa logica dovrebbe essere più robusta.
        grouped_members AS (
            SELECT m1 AS model, COLLECT_SET(m2) AS related_models FROM similar_pairs GROUP BY m1
            UNION ALL
            SELECT m2 AS model, COLLECT_SET(m1) AS related_models FROM similar_pairs GROUP BY m2
        ),
        -- Unire i set per creare gruppi più ampi basati sulla connettività
        -- Questo è un'approssimazione; per una soluzione precisa di Connected Components,
        -- si userebbe un algoritmo iterativo o GraphFrames.
        -- Per semplicità e aderenza a SQL, aggreghiamo per la prima colonna disponibile.
        final_groups AS (
            SELECT model, COLLECT_SET(exploded_member) as members
            FROM grouped_members
            LATERAL VIEW EXPLODE(related_models) exploded_table as exploded_member
            GROUP BY model
        ),
        all_models_with_group_id AS (
            SELECT model, array_sort(members) as group_id_array FROM final_groups
        ),
        -- Calcolo statistiche per i membri del gruppo
        group_stats AS (
          SELECT ag.group_id_array,
                 c.model_name,
                 AVG(c.price)   AS avg_price_member,
                 AVG(c.horsepower) AS avg_hp_member
          FROM all_models_with_group_id ag
          JOIN cars c
            ON array_contains(ag.group_id_array, c.model_name) -- un modello appartiene a un gruppo se è nel suo group_id_array
          GROUP BY ag.group_id_array, c.model_name
        ),
        -- CTE per trovare il modello con potenza media massima in ciascun gruppo
        ranked AS (
          SELECT
            group_id_array,
            model_name,
            avg_price_member,
            avg_hp_member,
            ROW_NUMBER() OVER (
              PARTITION BY group_id_array
              ORDER BY avg_hp_member DESC, model_name ASC -- Aggiunto model_name per ordine deterministico
            ) AS rn
          FROM group_stats
        ),
        -- CTE per aggregare membri e prezzo medio del gruppo
        agg AS (
          SELECT
            group_id_array,
            array_sort(COLLECT_SET(model_name)) AS group_members,
            ROUND(AVG(avg_price_member), 2) AS group_avg_price
          FROM group_stats
          GROUP BY group_id_array
        )
        SELECT
          a.group_id_array[0] AS group_representative_model, -- Prendo il primo modello del group_id_array come rappresentante
          a.group_members,
          a.group_avg_price,
          r.model_name AS top_power_model
        FROM agg a
        JOIN ranked r
          ON a.group_id_array = r.group_id_array -- Join sui group_id_array completi
        AND r.rn = 1
        ORDER BY group_representative_model;
        """
        # CRITICITÀ: Il calcolo dei "gruppi" (componenti connesse) in SQL è intrinsecamente difficile
        # se la similarità può estendersi (A-B, B-C implica A-C).
        # La logica sopra cerca di approssimare raccogliendo tutti i membri correlati, ma
        # non garantisce gruppi disgiunti o componenti connesse complete senza iterazione.
        # Per un problema Graph-like, GraphFrames sarebbe l'ideale.

        start = time.time()
        result_df = spark.sql(sql)

        # 🔸 trigger + stampa su console del driver
        # Forzo l'esecuzione per misurare il tempo
        sample_results = result_df.limit(10).collect()
        duration = round(time.time() - start, 2)

        # 🔸 log su file
        f_out.write("Prime 10 risultati:\n")
        sys.stdout.write("Prime 10 risultati:\n")
        for row in sample_results:
            # Formattazione per la stampa
            line = f"Group Rep: {row.group_representative_model}, Members: {row.group_members}, Avg Price: {row.group_avg_price}, Top Power Model: {row.top_power_model}"
            sys.stdout.write(line + "\n")
            f_out.write(line + "\n")

        f_out.write(f"Tempo model_similarity Spark SQL: {duration} sec\n")
        sys.stdout.write(f"Tempo model_similarity Spark SQL: {duration} sec\n\n")

        exec_times.append(duration)
        labels.append(label)

        # 🗑️ Rimuovi dalla cache alla fine di ogni iterazione per liberare memoria
        df_clean.unpersist()

# Salviamo i tempi di esecuzione in un file separato per il grafico.
# Questo file sarà scaricato localmente per generare il plot.
with open(LOCAL_TIMES_FILE_ON_EMR, "w", encoding="utf-8") as tf:
    for i in range(len(labels)):
        tf.write(f"{labels[i]} {exec_times[i]}\n")

# Termina la sessione Spark
spark.stop()

sys.stdout.write(f"\n✅ Report salvato in: {LOCAL_LOG_FILE}\n")
sys.stdout.write(f"✅ Tempi di esecuzione salvati in: {LOCAL_TIMES_FILE_ON_EMR}\n")
sys.stdout.write(f"Per generare il grafico, scarica '{LOCAL_TIMES_FILE_ON_EMR}' e usa uno script Python locale.\n")