from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, regexp_extract
import time, os, re
import sys  # Importa sys per output su stderr/stdout del driver

# --- Configurazione per Cluster EMR ---
# Su EMR, i percorsi devono essere S3, non locali.
S3_INPUT_BASE_PATH = "s3://bucketpoggers2/input/"

# Gli output finali (se volessi scriverli) andrebbero su S3.
# Per questo script, i risultati vengono solo campionati e loggati.
S3_OUTPUT_BASE_PATH = "s3://bucketpoggers2/output/spark_city_priceband/"

# Il file di log e il file dei tempi verranno generati localmente sul nodo master EMR
# e poi scaricati.
LOCAL_LOG_DIR = "/home/hadoop/spark_logs_city_priceband/"  # Directory sul nodo master EMR per i log temporanei
LOCAL_LOG_FILE = os.path.join(LOCAL_LOG_DIR, "spark_core_city_priceband.txt")
LOCAL_TIMES_FILE_ON_EMR = os.path.join(LOCAL_LOG_DIR, "spark_core_city_priceband_times.txt")

# Assicurati che la directory di log esista sul nodo driver EMR
os.makedirs(LOCAL_LOG_DIR, exist_ok=True)

# ⚡ Inizializza SparkSession. Senza configurazioni esplicite.
spark = SparkSession.builder \
    .appName("SparkCityPriceBandCluster") \
    .getOrCreate()

# Imposta il livello di log per avere un output più pulito
spark.sparkContext.setLogLevel("ERROR")
sc = spark.sparkContext  # Ottieni SparkContext per operazioni RDD

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
    ('used_cars_data_1_5x.csv', '150%'),  # Aggiunto 1.5x dal precedente se vuoi mantenerlo
    ('used_cars_data_2x.csv', '200%')
]

exec_times = []
labels = []

# stopwords facoltative (saranno serializzate e inviate ai workers)
stopwords = set(["and", "the", "a", "in", "of", "to", "for", "on", "with", "our", "your"])


# Funzione per pulire le parole (sarà serializzata e inviata ai workers)
def clean_word(w):
    s = re.sub(r'^\W+|\W+$', '', w)
    return s


with open(LOCAL_LOG_FILE, "w", encoding="utf-8") as fout:
    for file_name, label in datasets:
        s3_path = S3_INPUT_BASE_PATH + file_name
        fout.write(f"\n== Dataset {label}: {s3_path} ==\n")
        sys.stdout.write(f"\n== Dataset {label}: {s3_path} ==\n")

        # --- 3) Leggi con Spark DataFrame, sfruttando reader robusto ---
        # Spark legge direttamente da S3.
        df = spark.read.option("header", True) \
            .option("inferSchema", True) \
            .option("multiLine", True) \
            .option("escape", "\\") \
            .option("quote", "\"") \
            .csv(s3_path)

        # --- 4) Estrai solo le colonne necessarie e puliscile ---
        # Utilizza regexp_extract per una pulizia più robusta dei campi numerici
        # e per assicurarsi che i campi city siano solo alfabetici.
        df2 = df.select(
            regexp_extract(col("city"), r"^[A-Za-zÀ-ÖØ-öø-ÿ' \-]{1,50}", 0).alias("city"),
            # Estrai solo caratteri validi
            col("year"),
            # Pulizia e cast del prezzo prima della logica di band
            when(regexp_extract(col("price"), r"[\d\.]+", 0).cast("float") > 50000, "high")
            .when((regexp_extract(col("price"), r"[\d\.]+", 0).cast("float") >= 20000) & \
                  (regexp_extract(col("price"), r"[\d\.]+", 0).cast("float") <= 50000), "medium")
            .otherwise("low")
            .alias("band"),
            regexp_extract(col("daysonmarket"), r"\d+", 0).cast("int").alias("daysonmarket"),  # Estrai solo numeri
            col("description")
        ).filter(
            # Filtra i valori puliti e convertiti
            (col("city").isNotNull()) & (col("city") != "") &  # Assicurati che city non sia nullo o vuoto dopo la regex
            (col("year").between(1900, 2025)) &
            (regexp_extract(col("price"), r"[\d\.]+", 0).cast(
                "float").isNotNull()) &  # Prezzo originale deve essere numerico
            (regexp_extract(col("price"), r"[\d\.]+", 0).cast("float") > 0)  # Prezzo deve essere positivo
        ).cache()  # Cache il DataFrame filtrato se usato più volte

        # --- 5) Converti in RDD di tuple chiave/valore ---
        # Assicurati che i tipi siano corretti dopo le operazioni DataFrame.
        # r.daysonmarket è già int dal DataFrame, r.year è già int.
        rdd = df2.rdd.map(lambda r: (
            # chiave: (city, year, band)
            (r.city,
             r.year,
             r.band),
            # valore: (count=1, daysonmarket, description)
            (1,
             r.daysonmarket or 0,  # Default a 0 se daysonmarket è null (dopo cast a int, improbabile ma per sicurezza)
             r.description or "")
        ))

        # --- 6) Statistiche numeriche (Spark Core) ---
        # group by (city, year, band)
        grouped_stats = rdd.mapValues(lambda v: (v[0], v[1])) \
            .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1])) \
            .mapValues(lambda t: (t[0], round(t[1] / t[0], 2))) \
            .cache()  # Cache questo RDD intermedio

        # --- 7) Conteggio parole pulite (su tutte le descrizioni) ---
        # La funzione clean_word e la set stopwords saranno serializzate e inviate agli executor.
        words = rdd.flatMap(lambda x: [
            # ((city, year, band, cleaned_word), 1)
            ((x[0][0], x[0][1], x[0][2], cw), 1)
            for w in x[2].split()  # x[2] è r.description dalla lambda del rdd
            if (cw := clean_word(w.lower())).isalpha() and cw not in stopwords
        ])
        wc = words.reduceByKey(lambda a, b: a + b) \
            .cache()  # Cache il conteggio parole intermedio

        # --- 8) Top‑3 parole per ogni (city, year, band) ---
        top3 = wc.map(lambda x: ((x[0][0], x[0][1], x[0][2]), (x[0][3], x[1]))) \
            .groupByKey() \
            .mapValues(lambda seq: [w for w, _ in sorted(seq, key=lambda x: -x[1])[:3]])

        # --- 9) Benchmark join e take(10) ---
        start = time.time()
        joined = grouped_stats.join(top3)
        sample = joined.take(10)  # take() forza l'esecuzione
        duration = round(time.time() - start, 2)

        # --- 10) Output ---
        fout.write("Prime 10 risultati:\n")
        sys.stdout.write("Prime 10 risultati:\n")
        for (c, y, b), ((cnt, avgd), topw) in sample:
            line = f"{c} | {y} | {b} | count={cnt} | avg_days={avgd} | top3={topw}"
            sys.stdout.write(line + "\n")
            fout.write(line + "\n")

        fout.write(f"Tempo aggregation: {duration} sec\n")
        sys.stdout.write(f"Tempo aggregation: {duration} sec\n")

        exec_times.append(duration)
        labels.append(label)

        # Rilascia la cache dei RDD/DataFrame per liberare memoria
        df2.unpersist()
        grouped_stats.unpersist()
        wc.unpersist()

# Scrivi i tempi di esecuzione in un file separato sul nodo master EMR
with open(LOCAL_TIMES_FILE_ON_EMR, "w", encoding="utf-8") as tf:
    for i in range(len(labels)):
        tf.write(f"{labels[i]} {exec_times[i]}\n")

# 11) Arresta SparkSession
spark.stop()

sys.stdout.write(f"\n✅ Report salvato in: {LOCAL_LOG_FILE}\n")
sys.stdout.write(f"✅ Tempi di esecuzione salvati in: {LOCAL_TIMES_FILE_ON_EMR}\n")
sys.stdout.write(f"Per generare il grafico, scarica '{LOCAL_TIMES_FILE_ON_EMR}' e usa uno script Python locale.\n")