from pyspark.sql import SparkSession
import time
import csv
import os
import sys

# --- Configurazione per Cluster EMR ---
# Su EMR, i percorsi devono essere S3.
S3_INPUT_BASE_PATH = "s3://bucketpoggers2/input/"

# Gli output finali andranno su S3.
S3_OUTPUT_BASE_PATH = "s3://bucketpoggers2/output/spark_core_AVG/"

# Il file di log e il grafico verranno generati localmente sul nodo master EMR
# e poi scaricati.
LOCAL_LOG_DIR = "/home/hadoop/spark_logs_simplified/"  # Directory sul nodo master EMR per i log temporanei
LOCAL_LOG_FILE = os.path.join(LOCAL_LOG_DIR, "spark_core_model_stats.txt")
LOCAL_TIMES_FILE_ON_EMR = os.path.join(LOCAL_LOG_DIR, "spark_core_times.txt")

# Assicurati che la directory di log esista sul nodo driver EMR
os.makedirs(LOCAL_LOG_DIR, exist_ok=True)

# ⚡ Inizializza SparkSession.
# Non sono più necessarie le configurazioni esplicite qui, Spark le gestirà.
spark = SparkSession.builder \
    .appName("SparkCoreModelStatsClusterSimplified") \
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
    ('used_cars_data_2x.csv', '200%'),
    ('used_cars_data_4x.csv', '400%')
]

exec_times = []
labels = []

# 3) Apri file di log sul nodo master EMR
with open(LOCAL_LOG_FILE, "w", encoding="utf-8") as fout:
    for file_name, label in datasets:
        s3_path = S3_INPUT_BASE_PATH + file_name
        header = f"\n== Dataset {label}: {s3_path} ==\n"
        sys.stdout.write(header)
        fout.write(header)

        # a) leggi e scarta header
        rdd = sc.textFile(s3_path)
        first = rdd.first()
        data = rdd.filter(lambda row: row != first)


        # b) parsing & filtraggio
        def parse_line(line):
            try:
                fields = next(csv.reader([line]))
                make = fields[42].strip() if len(fields) > 42 and fields[42] else None
                model = fields[45].strip() if len(fields) > 45 and fields[45] else None

                price_str = fields[48].replace('$', '').replace(',', '').strip() if len(fields) > 48 and fields[
                    48] else ''
                price = float(price_str) if price_str else -1.0

                year_str = fields[65].strip() if len(fields) > 65 and fields[65] else ''
                year = int(year_str) if year_str else -1

                if not make or not model:      return None
                if price <= 0 or year < 1900 or year > 2025: return None
                return (make, model, price, year)
            except Exception as e:
                sys.stderr.write(f"Error parsing line: {line.strip()}. Error: {e}\n")
                return None


        parsed = data.map(parse_line).filter(lambda x: x is not None).cache()

        # c) prepara KV per reduceByKey
        kv = parsed.map(lambda x: (
            (x[0], x[1]),
            (1, x[2], x[2], x[2], {x[3]})
        ))


        # d) funzione di combinazione
        def combine(a, b):
            return (
                a[0] + b[0],
                min(a[1], b[1]),
                max(a[2], b[2]),
                a[3] + b[3],
                a[4].union(b[4])
            )


        # e) misura tempo reduceByKey + take
        start = time.time()
        aggregated = kv.reduceByKey(combine)

        sample = aggregated.take(10)
        duration = round(time.time() - start, 2)

        # f) stampa a schermo e log
        fout.write("Prime 10 risultati aggregati:\n")
        sys.stdout.write("\nPrime 10 risultati aggregati:\n")
        for (make, model), stats in sample:
            count = stats[0]
            min_p = stats[1]
            max_p = stats[2]
            avg_p = round(stats[3] / count, 2) if count > 0 else 0.0
            years = sorted(list(stats[4]))
            line = f"make={make} | model={model} | count={count} | min={min_p} | max={max_p} | avg={avg_p} | years={years}"
            sys.stdout.write(line + "\n")
            fout.write(line + "\n")

        timing_line = f"Spark Core Aggregation Time: {duration} sec\n"
        sys.stdout.write(timing_line)
        fout.write(timing_line)

        exec_times.append(duration)
        labels.append(label)

        parsed.unpersist()

# Scrivi i tempi di esecuzione in un file separato sul nodo master EMR
with open(LOCAL_TIMES_FILE_ON_EMR, "w", encoding="utf-8") as tf:
    for i in range(len(labels)):
        tf.write(f"{labels[i]} {exec_times[i]}\n")

# 4) Arresta SparkSession
spark.stop()

sys.stdout.write(f"\n✅ Log delle tabelle salvato in: {LOCAL_LOG_FILE}\n")
sys.stdout.write(f"✅ Tempi di esecuzione salvati in: {LOCAL_TIMES_FILE_ON_EMR}\n")
sys.stdout.write(
    f"Per generare il grafico, scarica '{LOCAL_TIMES_FILE_ON_EMR}' e usa lo script 'generate_graph.py' localmente.\n")