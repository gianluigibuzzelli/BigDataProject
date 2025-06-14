from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, regexp_replace, isnan, split, explode, lower, count, avg, array_sort
import time
import os
import sys  # Importa sys per output su stderr/stdout del driver


# --- Configurazione per Cluster EMR ---
# Su EMR, i percorsi devono essere S3, non locali.
S3_INPUT_BASE_PATH = "s3://bucketpoggers2/input/"  # Assicurati che i tuoi file siano qui

# Il file di log e il file dei tempi verranno generati localmente sul nodo master EMR
LOCAL_LOG_DIR = "/home/hadoop/spark_logs_city_priceband/"  # Directory sul nodo master EMR per i log temporanei
LOCAL_LOG_FILE = os.path.join(LOCAL_LOG_DIR, "spark_sql_city_priceband_results.txt")
LOCAL_TIMES_FILE_ON_EMR = os.path.join(LOCAL_LOG_DIR, "spark_sql_city_priceband_exec_times.txt")

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
    ('used_cars_data_2x.csv', '200%')
]

# ⚡ Inizializza SparkSession. Senza configurazioni esplicite.
# Spark userà le configurazioni di default del cluster EMR.
spark = SparkSession.builder \
    .appName("CityYearPriceBandBenchmarkCluster") \
    .getOrCreate()

# Imposta il livello di log per avere un output più pulito
spark.sparkContext.setLogLevel("ERROR")

essential = ['make_name', 'model_name', 'horsepower', 'engine_displacement']
int_cols = ['daysonmarket', 'dealer_zip', 'listing_id', 'owner_count', 'maximum_seating', 'year']

exec_times = []
labels = []

# 🔸 Prepara directory e file di output (su nodo master EMR)
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

        # Pulizia base e casting dei tipi
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
            df_clean = df_clean.withColumn(c, when(
                col(c).isNull() | isnan(col(c)) | ~col(c).cast("string").rlike("^-?\\d+$"), -1).otherwise(
                col(c).cast("int")))

        # Gestione valori nulli per 'description'
        df_clean = df_clean.withColumn("description",
                                       when(col("description").isNull(), "").otherwise(col("description")))

        # Filtri precoci per ridurre la quantità di dati il prima possibile
        df_filtered = df_clean.filter(
            (col("price") > 0) &
            (col("price") < 1_000_000) &
            (col("year") >= 1900) &
            (col("year") <= 2025)
        ).filter(~col("make_name").rlike("^[0-9]+(\\.[0-9]+)? in$"))  # Rimuove 'make_name' come '123 in'

        # Pulizia city e filtro per ridurre ulteriormente le righe
        # Assicurati che la colonna 'city' esista prima di applicare i filtri
        if "city" in df_filtered.columns:
            df_filtered = df_filtered.filter(col("city").isNotNull() & (col("city").rlike("^.{1,50}$")))
            df_filtered = df_filtered.filter(col("city").rlike("^[A-Za-zÀ-ÖØ-öø-ÿ'\\- ]+$"))
        else:
            sys.stdout.write(
                f"AVVISO: Colonna 'city' non trovata in {s3_path}. Saltando i filtri specifici per 'city'.\n")
            f_out.write(f"AVVISO: Colonna 'city' non trovata in {s3_path}. Saltando i filtri specifici per 'city'.\n")

        # 🚀 Ottimizzazione: Caching del DataFrame pulito e filtrato
        # Importante per query complesse che riutilizzano lo stesso DataFrame base.
        df_filtered.cache()
        # Forza il calcolo per popolare la cache. Utile in fase di benchmark.
        df_filtered.count()

        if df_filtered.head(1) == []:
            sys.stdout.write(
                f"AVVISO: Nessun dato valido dopo pulizia e filtro per {s3_path}. Salto questo dataset.\n\n")
            f_out.write(f"AVVISO: Nessun dato valido dopo pulizia e filtro per {s3_path}. Salto questo dataset.\n\n")
            df_filtered.unpersist()
            continue

        df_filtered.createOrReplaceTempView("cars")

        # Query SQL (invariata)
        # Nota: l'uso di LATERAL VIEW EXPLODE richiede che 'description' sia presente e pulita.
        sql = """
        WITH base AS (
          SELECT city, year, daysonmarket,
            CASE
              WHEN price > 50000 THEN 'high'
              WHEN price BETWEEN 20000 AND 50000 THEN 'medium'
              ELSE 'low'
            END AS price_band,
            description
          FROM cars
        ),
        agg_stats AS (
          SELECT city, year, price_band,
                 COUNT(*)                   AS num_auto,
                 ROUND(AVG(daysonmarket),2) AS avg_days_on_market
          FROM base
          GROUP BY city, year, price_band
        ),
        exploded AS (
          SELECT city, year, price_band, LOWER(word) AS word
          FROM base
          LATERAL VIEW EXPLODE(SPLIT(description, '\\\\W+')) AS word
          WHERE word <> ''
        ),
        word_counts AS (
          SELECT city, year, price_band, word, COUNT(*) AS wcount,
                 ROW_NUMBER() OVER (PARTITION BY city, year, price_band ORDER BY COUNT(*) DESC) AS rn
          FROM exploded
          GROUP BY city, year, price_band, word
        ),
        top_words AS (
          SELECT city, year, price_band, COLLECT_LIST(word) AS top3_words
          FROM word_counts
          WHERE rn <= 3
          GROUP BY city, year, price_band
        )
        SELECT a.city, a.year, a.price_band, a.num_auto, a.avg_days_on_market, array_sort(w.top3_words) as top3_words_sorted
        FROM agg_stats a
        JOIN top_words w
          ON a.city = w.city AND a.year = w.year AND a.price_band = w.price_band
        ORDER BY a.city, a.year, a.price_band
        """

        start = time.time()
        result_df = spark.sql(sql)
        # La chiamata a .count() forza l'esecuzione della query per ottenere un tempo reale
        result_df.count()
        duration = round(time.time() - start, 2)

        # Mostra solo i primi 10 risultati per evitare di riempire la console e la RAM con l'output completo
        sample_results = result_df.limit(10).collect()

        f_out.write("Prime 10 risultati:\n")
        sys.stdout.write("Prime 10 risultati:\n")
        for row in sample_results:
            # Formattazione per la stampa, assicurati che la lista di parole sia ordinata per consistenza
            line = f"City: {row.city}, Year: {row.year}, Price Band: {row.price_band}, Auto Count: {row.num_auto}, Avg Days On Market: {row.avg_days_on_market}, Top 3 Words: {row.top3_words_sorted}"
            sys.stdout.write(line + "\n")
            f_out.write(line + "\n")

        sys.stdout.write(f"Tempo job city–year–price_band: {duration} sec\n")
        f_out.write(f"Tempo job city–year–price_band: {duration} sec\n\n")

        exec_times.append(duration)
        labels.append(label)

        # 🗑️ Rimuovi dalla cache alla fine di ogni iterazione per liberare memoria
        df_filtered.unpersist()

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