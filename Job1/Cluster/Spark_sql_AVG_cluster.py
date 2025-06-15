from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, regexp_replace, isnan
import time
import os
import sys  # Importa sys per output su stderr/stdout del driver

# --- Configurazione per Cluster EMR ---
# Su EMR, i percorsi devono essere S3, non locali.
S3_INPUT_BASE_PATH = "s3://bucketpoggers2/input/"  

# Il file di log e il file dei tempi verranno generati localmente sul nodo master EMR
LOCAL_LOG_DIR = "/home/hadoop/spark_logs_sql_make_model/"  # Directory sul nodo master EMR per i log temporanei
LOCAL_LOG_FILE = os.path.join(LOCAL_LOG_DIR, "spark_sql_results.txt")
LOCAL_TIMES_FILE_ON_EMR = os.path.join(LOCAL_LOG_DIR, "spark_sql_exec_times.txt")  # File per i tempi

# Assicurati che la directory di log esista sul nodo driver EMR
os.makedirs(LOCAL_LOG_DIR, exist_ok=True)

# 🔧 Impostazioni
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
    ('used_cars_data_2x.csv', '200%'),
    ('used_cars_data_4x.csv', '400%')
]

# ⚡ Inizializza SparkSession. Senza configurazioni esplicite.
# Spark userà le configurazioni di default del cluster EMR.
spark = SparkSession.builder \
    .appName("UsedCarsSparkSQLStatsCluster") \
    .getOrCreate()

# Imposta il livello di log per avere un output più pulito
spark.sparkContext.setLogLevel("ERROR")

essential = ['make_name', 'model_name', 'horsepower', 'engine_displacement']
int_cols = ['daysonmarket', 'dealer_zip', 'listing_id', 'owner_count', 'maximum_seating', 'year']

exec_times = []
labels = []

# 📁 File output (su nodo master EMR)
with open(LOCAL_LOG_FILE, "w", encoding="utf-8") as f_out:
    for file_name, label in datasets:
        s3_path = S3_INPUT_BASE_PATH + file_name
        f_out.write(f"\n== Analisi Dataset: {label} ({s3_path}) ==\n")
        sys.stdout.write(f"\n== Analisi Dataset: {label} ({s3_path}) ==\n")

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

        # Pulizia e preparazione dati (stessa logica del tuo script originale)
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
            df_clean = df_clean.withColumn(c, when(col(c).isNull() | isnan(col(c)), -1).otherwise(col(c).cast("int")))

        # Gestione valori nulli per 'description'
        df_clean = df_clean.withColumn("description",
                                       when(col("description").isNull(), "").otherwise(col("description")))

        # Filtri finali di validità
        df_clean = df_clean.filter(
            (col("price") > 0) &
            (col("price") < 1_000_000) &
            (col("year") >= 1900) & (col("year") <= 2025)
        ).filter(~col("make_name").rlike("^[0-9]+(\\.[0-9]+)? in$"))  # Rimuove 'make_name' come '123 in'

        # Cache il DataFrame pulito per prestazioni migliori se riutilizzato
        df_clean.cache()

        if df_clean.head(1) == []:
            sys.stdout.write(
                f"AVVISO: Nessun dato valido dopo pulizia e filtro per {s3_path}. Salto questo dataset.\n\n")
            f_out.write(f"AVVISO: Nessun dato valido dopo pulizia e filtro per {s3_path}. Salto questo dataset.\n\n")
            df_clean.unpersist()
            continue

        df_clean.createOrReplaceTempView("cars")

        query = """
        SELECT
            make_name,
            model_name,
            COUNT(*) AS model_count,
            MIN(price) AS min_price,
            MAX(price) AS max_price,
            ROUND(AVG(price), 2) AS avg_price,
            COLLECT_SET(year) AS years_available
        FROM cars
        GROUP BY make_name, model_name
        ORDER BY make_name, model_name
        """

        start = time.time()
        result_df = spark.sql(query)

        # result_df.show(10, truncate=False) # Non chiamare show() direttamente, altrimenti stampa sul driver log
        sample_results = result_df.limit(10).collect()  # Prendi un campione per stampa/log
        duration = round(time.time() - start, 2)

        f_out.write("Prime 10 risultati:\n")
        sys.stdout.write("Prime 10 risultati:\n")
        for row in sample_results:
            line = f"Make: {row.make_name}, Model: {row.model_name}, Count: {row.model_count}, Min Price: {row.min_price}, Max Price: {row.max_price}, Avg Price: {row.avg_price}, Years: {sorted(list(row.years_available))}"
            sys.stdout.write(line + "\n")
            f_out.write(line + "\n")

        sys.stdout.write(f"Tempo analisi Spark SQL: {duration} sec\n")
        f_out.write(f"Tempo analisi Spark SQL: {duration} sec\n\n")

        exec_times.append(duration)
        labels.append(label)

        df_clean.unpersist()  # Rilascia la cache del DataFrame per liberare memoria

# Scrivi i tempi di esecuzione in un file separato sul nodo master EMR
with open(LOCAL_TIMES_FILE_ON_EMR, "w", encoding="utf-8") as tf:
    for i in range(len(labels)):
        tf.write(f"{labels[i]} {exec_times[i]}\n")

# 📊 Termina la SparkSession
spark.stop()

sys.stdout.write(f"\n✅ Report salvato in: {LOCAL_LOG_FILE}\n")
sys.stdout.write(f"✅ Tempi di esecuzione salvati in: {LOCAL_TIMES_FILE_ON_EMR}\n")
sys.stdout.write(f"Per generare il grafico, scarica '{LOCAL_TIMES_FILE_ON_EMR}' e usa uno script Python locale.\n")