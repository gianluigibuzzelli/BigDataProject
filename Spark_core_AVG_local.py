from pyspark.sql import SparkSession
import time, csv, os
import matplotlib.pyplot as plt
import numpy as np

SPARK_LOCAL_DIR = "/media/gianluigi/Z Slim/"
os.makedirs(SPARK_LOCAL_DIR, exist_ok=True) # Crea la directory se non esiste
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
    ('/media/gianluigi/Z Slim/DownloadUbuntu/archive/used_cars_data_2x.csv', '200%'),
    ('/media/gianluigi/Z Slim/DownloadUbuntu/archive/used_cars_data_4x.csv', '400%')
]

# ⚡ Inizializza SparkSession con le configurazioni ottimizzate
spark_builder = SparkSession.builder \
    .appName("AverageSparkCore")

for k, v in spark_configs.items():
    spark_builder = spark_builder.config(k, v)

spark = spark_builder.getOrCreate()

# Imposta il livello di log per avere un output più pulito
spark.sparkContext.setLogLevel("ERROR")
sc = spark.sparkContext

output_dir  = "/media/gianluigi/Z Slim/Risultati_locale/"
log_file    = os.path.join(output_dir, "spark_core_model_stats.txt")
graph_file  = os.path.join(output_dir, "spark_core_model_stats_times.png")

os.makedirs(output_dir, exist_ok=True)

exec_times = []
labels     = []

# 3) Apri file di log
with open(log_file, "w", encoding="utf-8") as fout:

    for path, label in datasets:
        header = f"\n== Dataset {label}: {path} ==\n"
        print(header.strip())
        fout.write(header)

        # a) leggi e scarta header
        rdd    = sc.textFile(path)
        first  = rdd.first()
        data   = rdd.filter(lambda row: row != first)

        # b) parsing & filtraggio
        def parse_line(line):
            try:
                fields = next(csv.reader([line]))
                make  = fields[42] or None
                model = fields[45] or None
                price = float(fields[48].replace('$','').replace(',','')) if fields[48] else -1.0
                year  = int(fields[65]) if fields[65] else -1
                if not make or not model:      return None
                if price <= 0 or year<1900 or year>2025: return None
                return (make, model, price, year)
            except:
                return None

        parsed = data.map(parse_line).filter(lambda x: x is not None)

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
        start      = time.time()
        aggregated = kv.reduceByKey(combine)
        sample     = aggregated.take(10)  # forza esecuzione
        duration   = round(time.time() - start, 2)

        # f) stampa a schermo e log
        fout.write("Prime 10 risultati aggregati:\n")
        print("\nPrime 10 risultati aggregati:")
        for (make, model), stats in sample:
            count    = stats[0]
            min_p    = stats[1]
            max_p    = stats[2]
            avg_p    = round(stats[3] / count, 2)
            years    = sorted(stats[4])
            line     = f"make={make} | model={model} | count={count} | min={min_p} | max={max_p} | avg={avg_p} | years={years}"
            print(line)
            fout.write(line + "\n")

        timing_line = f"Tempo aggregazione Spark Core: {duration} sec\n"
        print(timing_line.strip())
        fout.write(timing_line)

        exec_times.append(duration)
        labels.append(label)

# 4) Disegna e salva il grafico
plt.figure(figsize=(8,5))
plt.plot(labels, exec_times, marker='o', linestyle='-')
plt.title("Spark Core: Tempo aggregazione vs Dimensione dataset")
plt.xlabel("Dimensione dataset")
plt.ylabel("Tempo aggregazione (s)")
plt.ylim(0, max(exec_times) + 10)
plt.grid(True)
plt.tight_layout()
plt.savefig(graph_file)
plt.close()

print(f"\n✅ Log delle tabelle salvato in: {log_file}")
print(f"✅ Grafico dei tempi salvato in: {graph_file}")
