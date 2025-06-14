from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
import time, os, re
import matplotlib.pyplot as plt
import numpy as np

SPARK_LOCAL_DIR = "/media/gianluigi/Z Slim/Spark_tmp"
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
    ('/media/gianluigi/Z Slim/DownloadUbuntu/archive/used_cars_data_2x.csv', '200%')
]

# ⚡ Inizializza SparkSession con le configurazioni ottimizzate
spark_builder = SparkSession.builder \
    .appName("AverageSparkCore")

for k, v in spark_configs.items():
    spark_builder = spark_builder.config(k, v)

spark = spark_builder.getOrCreate()

# Imposta il livello di log per avere un output più pulito
spark.sparkContext.setLogLevel("ERROR")

output_dir  = "/media/gianluigi/Z Slim/Risultati_locale/"

os.makedirs(output_dir, exist_ok=True)
log_file   = os.path.join(output_dir, "spark_core_city_priceband.txt")
graph_file = os.path.join(output_dir, "spark_core_city_priceband_times.png")

exec_times = []
labels     = []

# stopwords facoltative
stopwords = set(["and","the","a","in","of","to","for","on","with","our","your"])

def clean_word(w):
    s = re.sub(r'^\W+|\W+$','',w)
    return s

with open(log_file, "w", encoding="utf-8") as fout:
    for path, label in datasets:
        fout.write(f"\n== Dataset {label}: {path} ==\n")
        print(f"\n== Dataset {label}: {path} ==")

        # --- 3) Leggi con Spark DataFrame, sfruttando reader robusto ---
        df = spark.read.option("header",True) \
                       .option("inferSchema",True) \
                       .option("multiLine",True) \
                       .option("escape","\\") \
                       .option("quote","\"") \
                       .csv(path)

        # --- 4) Estrai solo le colonne necessarie e puliscile ---
        df2 = df.select(
            col("city"),
            col("year"),
            when(col("price") > 50000, "high")
             .when((col("price") >= 20000) & (col("price") <= 50000), "medium")
             .otherwise("low")
             .alias("band"),
            col("daysonmarket"),
            col("description")
        ).filter(
            (col("price").isNotNull()) &
            (col("price") > 0) &
            (col("year").between(1900,2025)) &
            col("city").isNotNull()
        ).filter(
            col("city").rlike(r"^[A-Za-zÀ-ÖØ-öø-ÿ' \-]{1,50}$")
        )

        # --- 5) Converti in RDD di tuple chiave/valore ---
        # 5) Converti in RDD di tuple chiave/valore con safe cast
        rdd = df2.rdd.map(lambda r: (
            # chiave
            (r.city,
            int(float(r.year)),        # SAFE CAST da "2024.0" a 2024
            r.band),
            # valore
            (1,
            int(float(r.daysonmarket or 0)),
            r.description or "")
        ))

        # --- 6) Statistiche numeriche (Spark Core) ---
        grouped = rdd.mapValues(lambda v: (v[0], v[1])) \
                     .reduceByKey(lambda a, b: (a[0]+b[0], a[1]+b[1])) \
                     .mapValues(lambda t: (t[0], round(t[1]/t[0],2)))

        # --- 7) Conteggio parole pulite ---
        words = rdd.flatMap(lambda x: [
            ((x[0][0], x[0][1], x[0][2], clean_word(w.lower())), 1)
            for w in x[1][2].split()
            if (cw := clean_word(w.lower())).isalpha() and cw not in stopwords
        ])
        wc = words.reduceByKey(lambda a,b: a+b)

        # --- 8) Top‑3 parole ---
        top3 = wc.map(lambda x: ((x[0][0], x[0][1], x[0][2]), (x[0][3], x[1]))) \
                 .groupByKey() \
                 .mapValues(lambda seq: [w for w,_ in sorted(seq, key=lambda x:-x[1])[:3]])

        # --- 9) Benchmark join e take(10) ---
        start = time.time()
        joined = grouped.join(top3)
        sample = joined.take(10)
        duration = round(time.time() - start, 2)

        # --- 10) Output ---
        fout.write("Prime 10 risultati:\n")
        print("Prime 10 risultati:")
        for (c,y,b), ((cnt,avgd), topw) in sample:
            line = f"{c} | {y} | {b} | count={cnt} | avg_days={avgd} | top3={topw}"
            print(line)
            fout.write(line + "\n")

        fout.write(f"Tempo aggregation: {duration} sec\n")
        print(f"Tempo aggregation: {duration} sec")

        exec_times.append(duration)
        labels.append(label)

# 11) Grafico
plt.figure(figsize=(8,5))
plt.plot(labels, exec_times, marker='o')
plt.title("Spark Core City–Year–PriceBand: Tempo vs Dataset Size")
plt.xlabel("Sample size")
plt.ylabel("Execution time (s)")
plt.grid(True)
plt.tight_layout()
plt.savefig(graph_file)
plt.close()

print(f"\n✅ Report salvato in: {log_file}")
print(f"✅ Grafico salvato in: {graph_file}")
