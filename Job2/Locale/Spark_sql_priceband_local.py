from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, regexp_replace, isnan, split, explode, lower, count, avg
from pyspark.sql.window import Window
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

#Inizializza SparkSession con le configurazioni ottimizzate
spark_builder = SparkSession.builder \
    .appName("CityYearPriceBandBenchmarkWithCityClean")

for k, v in spark_configs.items():
    spark_builder = spark_builder.config(k, v)

spark = spark_builder.getOrCreate()

# Imposta il livello di log per avere un output più pulito
spark.sparkContext.setLogLevel("ERROR")

essential = ['make_name', 'model_name', 'horsepower', 'engine_displacement']
int_cols = ['daysonmarket', 'dealer_zip', 'listing_id', 'owner_count', 'maximum_seating', 'year']

exec_times = []
labels = []

# Prepara directory e file di output
os.makedirs("/media/gianluigi/Z Slim/Risultati_locale", exist_ok=True)
output_txt = "/media/gianluigi/Z Slim/Risultati_locale/spark_sql_city_priceband_results.txt"
graph_path = "/media/gianluigi/Z Slim/Risultati_locale/spark_sql_city_priceband_exec_times.png"

with open(output_txt, "w", encoding="utf-8") as f_out:
    for path, label in datasets:
        f_out.write(f"\n== Dataset {label}: {path} ==\n")
        print(f"\n== Dataset {label}: {path} ==")

        # Controllo esistenza file
        if not os.path.exists(path):
            print(f"ATTENZIONE: Il file non esiste al percorso: {path}. Saltando questo dataset.")
            f_out.write(f"ATTENZIONE: Il file non esiste al percorso: {path}. Saltando questo dataset.\n")
            continue

        df = spark.read.csv(path, header=True, inferSchema=True)

        # Pulizia base e casting dei tipi
        df = df.withColumn("price", regexp_replace("price", "[$,\\s]", "").cast("double"))
        df_clean = df.dropna(subset=essential)

        for c in int_cols:
            # Assicurati che le colonne int_cols non abbiano valori che impediscono la conversione in int
            # I valori non numerici o nulli vengono impostati a -1
            df_clean = df_clean.withColumn(c, when(
                col(c).isNull() | isnan(col(c)) | ~col(c).cast("string").rlike("^-?\\d+$"), -1).otherwise(
                col(c).cast("int")))

        df_clean = df_clean.withColumn("description",
                                       when(col("description").isNull(), "").otherwise(col("description")))

        # Filtri precoci per ridurre la quantità di dati il prima possibile
        df_filtered = df_clean.filter(
            (col("price") > 0) &
            (col("price") < 1_000_000) &
            (col("year") >= 1900) &
            (col("year") <= 2025)
        ).filter(~col("make_name").rlike("^[0-9]+(\\.[0-9]+)? in$"))

        # Pulizia city e filtro per ridurre ulteriormente le righe
        df_filtered = df_filtered.filter(col("city").isNotNull() & (col("city").rlike("^.{1,50}$")))
        df_filtered = df_filtered.filter(col("city").rlike("^[A-Za-zÀ-ÖØ-öø-ÿ'\\- ]+$"))



        df_filtered.createOrReplaceTempView("cars")

        # Query SQL invariata: è funzionalmente corretta e Spark la ottimizzerà con le nuove configurazioni
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
        SELECT a.city, a.year, a.price_band, a.num_auto, a.avg_days_on_market, w.top3_words
        FROM agg_stats a
        JOIN top_words w
          ON a.city = w.city AND a.year = w.year AND a.price_band = w.price_band
        ORDER BY a.city, a.year, a.price_band
        """

        start = time.time()
        result_df = spark.sql(sql)
        result_df.count()
        duration = round(time.time() - start, 2)

        # Mostra solo i primi 10 risultati per evitare di riempire la console e la RAM con l'output completo
        result_str = result_df._jdf.showString(10, 1000, False)  # Usa _jdf per accedere al metodo Java
        f_out.write(result_str + "\n")
        print("Tempo job city–year–price_band:", duration, "sec")
        f_out.write(f"Tempo job city–year–price_band: {duration} sec\n")

        exec_times.append(duration)
        labels.append(label)

        # 🗑️ Rimuovi dalla cache alla fine di ogni iterazione per liberare memoria
        #df_filtered.unpersist()

# 📊 Salva il grafico (codice invariato)
plt.figure(figsize=(10, 7))  # Aumenta la dimensione per una migliore leggibilità
plt.plot(labels, exec_times, marker='o', linestyle='-')
plt.title("Spark SQL City–Year–PriceBand: Tempo vs Dataset Size")
plt.xlabel("Sample size")
plt.ylabel("Execution time (s)")
plt.ylim(0, max(exec_times) + max(exec_times) * 0.1)  # Aggiunge un 10% di margine
plt.grid(True)
plt.tight_layout()
plt.savefig(graph_path)
plt.close()

print(f"\n✅ Report salvato in: {output_txt}")
print(f"✅ Grafico salvato in: {graph_path}")

# Termina la sessione Spark
spark.stop()