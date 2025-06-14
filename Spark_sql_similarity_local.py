from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, regexp_replace, isnan
import time
import matplotlib.pyplot as plt
import numpy as np
import os

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
    ('/media/gianluigi/Z Slim/DownloadUbuntu/archive/used_cars_data_2x.csv', '200%')
]

# ⚡ Inizializza SparkSession con le configurazioni ottimizzate
spark_builder = SparkSession.builder \
    .appName("SimilarityEngine")

for k, v in spark_configs.items():
    spark_builder = spark_builder.config(k, v)

spark = spark_builder.getOrCreate()

# Imposta il livello di log per avere un output più pulito
spark.sparkContext.setLogLevel("ERROR")

essential = ['make_name', 'model_name', 'horsepower', 'engine_displacement']
int_cols  = ['daysonmarket', 'dealer_zip', 'listing_id', 'owner_count', 'maximum_seating', 'year']

exec_times = []
labels     = []

# 📁 Prepara directory e file di output
os.makedirs("/media/gianluigi/Z Slim/Risultati_locale", exist_ok=True)
output_txt = "/media/gianluigi/Z Slim/Risultati_locale/model_similarity_results.txt"
graph_path = "/media/gianluigi/Z Slim/Risultati_locale/model_similarity_exec_times.png"

with open(output_txt, "w", encoding="utf-8") as f_out:

    for path, label in datasets:
        f_out.write(f"\n== Dataset {label}: {path} ==\n")
        print(f"\n== Dataset {label}: {path} ==")

        # --- lettura e pulizia identica ---
        df = spark.read.csv(path, header=True, inferSchema=True)
        df = df.withColumn("price", regexp_replace("price","[$,\\s]","").cast("double"))
        df_clean = df.dropna(subset=essential)
        for c in int_cols:
            df_clean = df_clean.withColumn(
                c,
                when(col(c).isNull() | isnan(col(c)), -1).otherwise(col(c).cast("int"))
            )
        df_clean = df_clean.withColumn(
            "description",
            when(col("description").isNull(), "").otherwise(col("description"))
        ).filter(
            (col("price")>0) &
            (col("price")<1_000_000) &
            (col("year")>=1900)&(col("year")<=2025)
        ).filter(~col("make_name").rlike("^[0-9]+(\\.[0-9]+)? in$"))

        df_clean.createOrReplaceTempView("cars")

        # --- similarity job in Spark SQL ---
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
            ON a.model_name <= b.model_name
          AND ABS(a.avg_hp - b.avg_hp)/a.avg_hp <= 0.10
          AND ABS(a.avg_ed - b.avg_ed)/a.avg_ed <= 0.10
        ),
        groups AS (
          SELECT m1 AS group_model, COLLECT_SET(m2) AS members
          FROM similar_pairs
          GROUP BY m1
        ),
        exploded AS (
          SELECT group_model, member
          FROM groups
          LATERAL VIEW EXPLODE(members) AS member
        ),
        group_stats AS (
          SELECT e.group_model,
                e.member       AS model_name,
                AVG(c.price)   AS avg_price_member,
                AVG(c.horsepower) AS avg_hp_member
          FROM exploded e
          JOIN cars c
            ON e.member = c.model_name
          GROUP BY e.group_model, e.member
        ),
        -- CTE per trovare il modello con potenza media massima in ciascun gruppo
        ranked AS (
          SELECT
            group_model,
            model_name,
            avg_price_member,
            avg_hp_member,
            ROW_NUMBER() OVER (
              PARTITION BY group_model
              ORDER BY avg_hp_member DESC
            ) AS rn
          FROM group_stats
        ),
        -- CTE per aggregare membri e prezzo medio del gruppo
        agg AS (
          SELECT
            group_model,
            COLLECT_SET(model_name) AS group_members,
            ROUND(AVG(avg_price_member), 2) AS group_avg_price
          FROM group_stats
          GROUP BY group_model
        )
        SELECT
          a.group_model,
          a.group_members,
          a.group_avg_price,
          r.model_name AS top_power_model
        FROM agg a
        JOIN ranked r
          ON a.group_model = r.group_model
        AND r.rn = 1
        ORDER BY a.group_model;

        """

        start = time.time()
        result_df = spark.sql(sql)

        # 🔸 trigger + stampa
        result_df.show(10, truncate=False)
        duration = round(time.time() - start, 2)

        # 🔸 log su file
        f_out.write(result_df._jdf.showString(10, 1000, False) + "\n")
        f_out.write(f"Tempo model_similarity Spark SQL: {duration} sec\n")

        print(f"Tempo model_similarity Spark SQL: {duration} sec")

        exec_times.append(duration)
        labels.append(label)

# --- plot dei tempi ---
plt.figure(figsize=(8,5))
plt.plot(labels, exec_times, marker='o', linestyle='-')
plt.title("Model Similarity Spark SQL: Tempo vs Dataset Size")
plt.xlabel("Sample size")
plt.ylabel("Execution time (s)")
plt.ylim(0, max(exec_times)+10)
plt.grid(True)
plt.tight_layout()
plt.savefig(graph_path)
plt.close()

print(f"\n✅ Report salvato in: {output_txt}")
print(f"✅ Grafico salvato in: {graph_path}")
