from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, regexp_replace, isnan
import time
import matplotlib.pyplot as plt
import numpy as np
import os

# 🔧 Impostazioni
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
    ('/media/gianluigi/Z Slim/DownloadUbuntu/archive/used_cars_data_2x.csv','200%'),
    ('/media/gianluigi/Z Slim/DownloadUbuntu/archive/used_cars_data_4x.csv','400%')
]
spark = SparkSession.builder \
    .appName("UsedCarsSparkSQLStats") \
    .getOrCreate()

essential = ['make_name', 'model_name', 'horsepower', 'engine_displacement']
int_cols = ['daysonmarket', 'dealer_zip', 'listing_id', 'owner_count', 'maximum_seating', 'year']

exec_times = []
labels = []

# 📁 File output
os.makedirs("/media/gianluigi/Z Slim/Risultati_locale", exist_ok=True)
output_txt = "/media/gianluigi/Z Slim/Risultati_locale/spark_sql_results.txt"
with open(output_txt, "w", encoding="utf-8") as f_out:
    for path, label in datasets:
        f_out.write(f"\n== Analisi Dataset: {label} ({path}) ==\n")
        print(f"\n== Analisi Dataset: {label} ({path}) ==")

        df = spark.read.csv(path, header=True, inferSchema=True)
        df = df.withColumn("price", regexp_replace("price", "[$,\\s]", "").cast("double"))
        df_clean = df.dropna(subset=essential)

        for c in int_cols:
            df_clean = df_clean.withColumn(c, when(col(c).isNull() | isnan(col(c)), -1).otherwise(col(c).cast("int")))

        df_clean = df_clean.withColumn("description", when(col("description").isNull(), "").otherwise(col("description")))

        df_clean = df_clean.filter(
            (col("price") > 0) &
            (col("price") < 1_000_000) &
            (col("year") >= 1900) & (col("year") <= 2025)
        ).filter(~col("make_name").rlike("^[0-9]+(\\.[0-9]+)? in$"))

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
        result_df.show(10, truncate=False)
        duration = round(time.time() - start, 2)
        result_str = result_df._jdf.showString(10, 1000, False)
        f_out.write(result_str + "\n")
        print("Tempo analisi Spark SQL:", duration, "sec")
        f_out.write(f"Tempo analisi Spark SQL: {duration} sec\n")

        exec_times.append(duration)
        labels.append(label)

# 📊 Salvataggio grafico
plt.figure(figsize=(8, 5))
plt.plot(labels, exec_times, marker='o', linestyle='-', color='blue')
plt.title("Tempo di esecuzione Spark SQL vs Dimensione Dataset")
plt.xlabel("Dimensione Dataset")
plt.ylabel("Tempo (secondi)")
y_max = max(exec_times)
plt.yticks(np.arange(0, y_max + 10, 10))
plt.grid(True)
plt.tight_layout()

graph_path = "/media/gianluigi/Z Slim/Risultati_locale/spark_sql_exec_times.png"
plt.savefig(graph_path)
plt.close()

print(f"\n✅ Report salvato in: {output_txt}")
print(f"✅ Grafico salvato in: {graph_path}")
