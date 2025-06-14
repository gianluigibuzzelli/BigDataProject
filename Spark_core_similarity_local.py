from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_replace, col, when, isnan
import time, os, csv
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
    ('/media/gianluigi/Z Slim/DownloadUbuntu/archive/used_cars_data_1_25x.csv', '125%')
]

# ⚡ Inizializza SparkSession con le configurazioni ottimizzate
spark_builder = SparkSession.builder \
    .appName("SimilaritySparkCore")

for k, v in spark_configs.items():
    spark_builder = spark_builder.config(k, v)

spark = spark_builder.getOrCreate()

# Imposta il livello di log per avere un output più pulito
spark.sparkContext.setLogLevel("ERROR")
sc = spark.sparkContext

output_dir  = "/media/gianluigi/Z Slim/Risultati_locale/"
os.makedirs(output_dir, exist_ok=True)
log_file   = os.path.join(output_dir, "model_similarity_core.txt")
graph_file = os.path.join(output_dir, "model_similarity_core_times.png")

exec_times = []
labels     = []

with open(log_file, "w", encoding="utf-8") as fout:
    for path, label in datasets:
        fout.write(f"\n== Dataset {label}: {path} ==\n")
        print(f"\n== Dataset {label}: {path} ==")

        # ---- 3) Leggi CSV con Spark DataFrame ----
        df = spark.read.option("header", True) \
                       .option("inferSchema", True) \
                       .option("multiLine", True) \
                       .option("quote", "\"") \
                       .option("escape", "\\") \
                       .csv(path)

        # ---- 4) Pulisci e filtra ----
        df_clean = df \
            .withColumn("price", regexp_replace("price","[$,\\s]","").cast("double")) \
            .withColumn("horsepower", col("horsepower").cast("double")) \
            .withColumn("engine_displacement", col("engine_displacement").cast("double")) \
            .dropna(subset=["model_name","horsepower","engine_displacement","price"]) \
            .filter((col("price")>0) &
                    (col("horsepower")>0) &
                    (col("engine_displacement")>0))

        # ---- 5) Converti in RDD[(model, (hp, ed, price))] ----
        rdd = df_clean.rdd.map(lambda r: (
            r.model_name,
            (r.horsepower, r.engine_displacement, r.price)
        ))

        # ---- 6) Calcola model_feat: RDD[(model, (avg_hp, avg_ed))] ----
        mf = (rdd
              .map(lambda x: (x[0], (x[1][0], x[1][1], 1)))
              .reduceByKey(lambda a,b: (a[0]+b[0], a[1]+b[1], a[2]+b[2]))
              .mapValues(lambda t: (t[0]/t[2], t[1]/t[2]))
              .cache()
        )
        model_feats = mf.collect()
        bc_feats    = sc.broadcast(model_feats)

        # ---- 7) Genera coppie simili O(M^2) ----
        def gen_pairs(_):
            out = []
            feats = bc_feats.value
            for i,(m1,(hp1,ed1)) in enumerate(feats):
                for m2,(hp2,ed2) in feats[i:]:
                    if abs(hp1-hp2)/hp1 <= 0.10 and abs(ed1-ed2)/ed1 <= 0.10:
                        out.append((m1, m2))
            return out

        similar = sc.parallelize([0]).flatMap(gen_pairs)

        # ---- 8) Raggruppa membri per each group_model ----
        groups = similar.groupByKey() \
                        .mapValues(lambda it: list(set(it))) \
                        .cache()

        # ---- 9) Prepara RDD per price e hp lookup ----
        prices = rdd.map(lambda x: (x[0], x[1][2]))
        hps    = rdd.map(lambda x: (x[0], x[1][0]))

        # ---- 10) Esplode: (member, group) per join ----
        exploded = groups.flatMap(lambda x: [(m, x[0]) for m in x[1]])

        # ---- 11) Calcola avg_price per group ----
        group_price = (exploded.join(prices)                       # (member, (group, price))
                       .map(lambda x: (x[1][0], (x[1][1], 1)))        # key=group
                       .reduceByKey(lambda a,b: (a[0]+b[0], a[1]+b[1]))
                       .mapValues(lambda t: round(t[0]/t[1],2))
        )  # RDD[(group, group_avg_price)]

        # ---- 12) Calcola top_power_model per group ----
        group_hp = (exploded.join(hps)                             # (member, (group, hp))
                    .map(lambda x: (x[1][0], (x[0], x[1][1])))       # (group, (member, hp))
                    .groupByKey()
                    .mapValues(lambda seq: max(seq, key=lambda y: y[1])[0])
        )  # RDD[(group, top_power_model)]

        # ---- 13) Unisci e misura tempi ----
        start = time.time()
        result = (group_price
                  .join(groups)     # (group, (avg_price, members))
                  .join(group_hp)   # (group, ((avg_price, members), top_model))
                  .map(lambda x: (
                      x[0],
                      x[1][0][1],    # members
                      x[1][0][0],    # avg_price
                      x[1][1]        # top_power_model
                  ))
                  .sortBy(lambda x: x[0])
        )
        sample   = result.take(10)
        duration = round(time.time() - start, 2)

        # ---- 14) Stampa e log ----
        fout.write("Prime 10 gruppi simili:\n")
        print("Prime 10 gruppi simili:")
        for grp, members, avgp, topm in sample:
            line = f"{grp} | members={members} | avg_price={avgp} | top_power_model={topm}"
            print(line)
            fout.write(line + "\n")

        print(f"Tempo ModelSimilarity Spark Core: {duration} sec")
        fout.write(f"Tempo ModelSimilarity Spark Core: {duration} sec\n\n")

        exec_times.append(duration)
        labels.append(label)

# ---- 15) Grafico dei tempi ----
plt.figure(figsize=(8,5))
plt.plot(labels, exec_times, marker='o', linestyle='-')
plt.title("Model Similarity Spark Core: Tempo vs Dataset Size")
plt.xlabel("Sample size")
plt.ylabel("Execution time (s)")
plt.ylim(0, max(exec_times)+10)
plt.grid(True)
plt.tight_layout()
plt.savefig(graph_file)
plt.close()

print(f"\n✅ Report e grafico salvati in: {output_dir}")
