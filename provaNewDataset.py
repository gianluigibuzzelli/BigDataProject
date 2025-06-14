import csv
from io import StringIO
import re
import time
import os
import matplotlib.pyplot as plt
import numpy as np
from pyspark.sql import SparkSession  # Mantenuto solo per inizializzare SparkSession, non per DataFrames

SPARK_LOCAL_DIR = "/media/gianluigi/Z Slim/Spark_tmp"
os.makedirs(SPARK_LOCAL_DIR, exist_ok=True)  # Crea la directory se non esiste
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

# La lista dei dataset è stata aggiornata per puntare ai file puliti nella directory 'cleaned'
datasets = [
    ('/media/gianluigi/Z Slim/DownloadUbuntu/cleaned/used_cars_data_sampled_1_cleaned.csv', '10%'),
    ('/media/gianluigi/Z Slim/DownloadUbuntu/cleaned/used_cars_data_sampled_2_cleaned.csv', '20%'),
    ('/media/gianluigi/Z Slim/DownloadUbuntu/cleaned/used_cars_data_sampled_3_cleaned.csv', '30%'),
    ('/media/gianluigi/Z Slim/DownloadUbuntu/cleaned/used_cars_data_sampled_4_cleaned.csv', '40%'),
    ('/media/gianluigi/Z Slim/DownloadUbuntu/cleaned/used_cars_data_sampled_5_cleaned.csv', '50%'),
    ('/media/gianluigi/Z Slim/DownloadUbuntu/cleaned/used_cars_data_sampled_6_cleaned.csv', '60%'),
    ('/media/gianluigi/Z Slim/DownloadUbuntu/cleaned/used_cars_data_sampled_7_cleaned.csv', '70%'),
    ('/media/gianluigi/Z Slim/DownloadUbuntu/cleaned/used_cars_data_sampled_8_cleaned.csv', '80%'),
    ('/media/gianluigi/Z Slim/DownloadUbuntu/cleaned/used_cars_data_sampled_9_cleaned.csv', '90%'),
    ('/media/gianluigi/Z Slim/DownloadUbuntu/cleaned/used_cars_data_cleaned.csv', '100%'),
    ('/media/gianluigi/Z Slim/DownloadUbuntu/cleaned/used_cars_data_1_25x_cleaned.csv', '125%'),
    ('/media/gianluigi/Z Slim/DownloadUbuntu/cleaned/used_cars_data_2x_cleaned.csv', '200%')
]

# ⚡ Inizializza SparkSession con le configurazioni ottimizzate
spark_builder = SparkSession.builder \
    .appName("AverageSparkCoreRDDOnly")  # Nome dell'applicazione modificato

for k, v in spark_configs.items():
    spark_builder = spark_builder.config(k, v)

spark = spark_builder.getOrCreate()

# Imposta il livello di log per avere un output più pulito
spark.sparkContext.setLogLevel("ERROR")
sc = spark.sparkContext  # Ottieni lo SparkContext per le operazioni RDD

output_dir = "/media/gianluigi/Z Slim/Risultati_locale/"
os.makedirs(output_dir, exist_ok=True)
# Nomi dei file di log e grafico aggiornati per riflettere l'uso RDD
log_file = os.path.join(output_dir, "spark_core_city_priceband_rdd_only.txt")
graph_file = os.path.join(output_dir, "spark_core_city_priceband_rdd_only_times.png")

exec_times = []
labels = []

# stopwords facoltative (mantienute per la tokenizzazione della descrizione)
stopwords = set(["and", "the", "a", "in", "of", "to", "for", "on", "with", "our", "your"])


def clean_word(w):
    """
    Pulisce una parola rimuovendo i caratteri non alfanumerici all'inizio e alla fine.
    """
    s = re.sub(r'^\W+|\W+$', '', w)
    return s


def parse_and_filter_csv_line(line: str, header_map: dict):
    """
    Parsa una singola riga CSV e applica i filtri e le conversioni di tipo.
    Args:
        line (str): La riga CSV da parsare.
        header_map (dict): Una mappa da nome colonna a indice.
    Returns:
        tuple: Una tupla nel formato ((city, year, band), (count, daysonmarket, description))
               o None se la riga è malformata o non supera i filtri.
    """
    try:
        # Usa csv.reader per gestire robustamente le virgolette e i delimitatori
        # Si assume che i file "puliti" siano ben formattati per csv.reader
        reader = csv.reader(StringIO(line), delimiter=',', doublequote=True, strict=True)
        fields = next(reader)

        # Verifica che la riga abbia il numero atteso di campi (basato sull'header)
        # Questo cattura righe tronche o con un numero di campi errato
        if len(fields) != len(header_map):
            return None

        # Estrazione e conversione dei campi necessari, con gestione degli errori
        city = fields[header_map['city']].strip() if 'city' in header_map and header_map['city'] < len(fields) else None

        year = None
        year_str = fields[header_map['year']].strip() if 'year' in header_map and header_map['year'] < len(
            fields) else None
        if year_str:
            try:
                year = int(float(year_str))  # Gestisce "2024.0" -> 2024
            except ValueError:
                pass  # year rimane None

        price = None
        price_str = fields[header_map['price']].strip() if 'price' in header_map and header_map['price'] < len(
            fields) else None
        if price_str:
            try:
                # Si assume che il file sia già pulito da valuta e spazi, ma per sicurezza...
                clean_price_str = re.sub(r'[^\d.]', '', price_str)
                price = float(clean_price_str)
            except ValueError:
                pass  # price rimane None

        daysonmarket = None
        daysonmarket_str = fields[header_map['daysonmarket']].strip() if 'daysonmarket' in header_map and header_map[
            'daysonmarket'] < len(fields) else None
        if daysonmarket_str:
            try:
                daysonmarket = int(float(daysonmarket_str))
            except ValueError:
                pass  # daysonmarket rimane None

        description = fields[header_map['description']].strip() if 'description' in header_map and header_map[
            'description'] < len(fields) else ""

        # Applicazione delle condizioni di filtro originali
        # Per la città, uso un regex più permissivo come suggerito in precedenza
        # per evitare di filtrare città come "St. Petersburg" dal dataset pulito.
        city_valid = city is not None and re.match(r"^[A-Za-zÀ-ÖØ-öø-ÿ0-9\s'\-,\.]{1,100}$", city.strip())

        if not (price is not None and price > 0 and
                year is not None and 1900 <= year <= 2025 and
                city_valid):
            return None  # La riga non supera i filtri

        # Calcolo della banda di prezzo (logica "when" del DataFrame)
        band = None
        if price is not None:
            if price > 50000:
                band = "high"
            elif 20000 <= price <= 50000:
                band = "medium"
            else:
                band = "low"
        else:  # Se il prezzo non è valido, non possiamo definire una banda
            return None

        # Restituisce la tupla nel formato desiderato per l'elaborazione RDD successiva
        return (
            (city, year, band),
            (1, daysonmarket if daysonmarket is not None else 0, description)
        )

    except (csv.Error, IndexError, StopIteration, ValueError) as e:
        # Cattura errori di parsing CSV, accesso a indici fuori bounds, righe vuote o conversioni
        return None
    except Exception as e:
        # Cattura qualsiasi altro errore imprevisto
        return None


with open(log_file, "w", encoding="utf-8") as fout:
    for path, label in datasets:
        fout.write(f"\n== Dataset {label}: {path} ==\n")
        print(f"\n== Dataset {label}: {path} ==")

        try:
            # --- 3) Leggi CSV con Spark RDD ---
            raw_rdd = sc.textFile(path)

            # Gestisce il caso di file vuoti o errori iniziali
            if raw_rdd.isEmpty():
                print(f"AVVISO: Il file {path} è vuoto o non accessibile. Salto questo dataset.")
                fout.write(f"AVVISO: Il file {path} è vuoto o non accessibile. Salto questo dataset.\n\n")
                continue

            header_line = raw_rdd.first()
            data_rdd = raw_rdd.filter(lambda line: line != header_line)

            # Parsa l'header per creare una mappa nome_colonna -> indice
            # Questo è cruciale per accedere ai campi per nome (es. 'city') e non per indice fisso
            header_fields = next(csv.reader(StringIO(header_line)))
            header_map = {name: idx for idx, name in enumerate(header_fields)}

            # Verifica che le colonne essenziali siano presenti
            required_cols = ['city', 'year', 'price', 'daysonmarket', 'description']
            if not all(col_name in header_map for col_name in required_cols):
                missing_cols = [col_name for col_name in required_cols if col_name not in header_map]
                print(
                    f"ERRORE: Colonne essenziali mancanti nell'header di {path}: {missing_cols}. Salto questo dataset.")
                fout.write(
                    f"ERRORE: Colonne essenziali mancanti nell'header di {path}: {missing_cols}. Salto questo dataset.\n\n")
                continue

        except Exception as e:
            print(
                f"ERRORE: Impossibile leggere o processare l'header del file {path}. Salto questo dataset. Errore: {e}")
            fout.write(
                f"ERRORE: Impossibile leggere o processare l'header del file {path}. Salto questo dataset. Errore: {e}\n\n")
            continue

        # --- 4) Applica parsing e filtraggio (sostituisce df.select e df.filter) ---
        # Le righe malformate o non conformi ai filtri sono restituite come None e poi rimosse
        rdd = data_rdd.map(lambda line: parse_and_filter_csv_line(line, header_map)).filter(lambda x: x is not None)

        # --- 5) Statistiche numeriche (Spark Core) ---
        grouped = rdd.mapValues(lambda v: (v[0], v[1])) \
            .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1])) \
            .mapValues(lambda t: (t[0], round(t[1] / t[0], 2)))

        # --- 6) Conteggio parole pulite ---
        words = rdd.flatMap(lambda x: [
            ((x[0][0], x[0][1], x[0][2], clean_word(w.lower())), 1)  # Key include city, year, band, cleaned_word
            for w in x[1][2].split()  # Access description from value tuple
            if (cw := clean_word(w.lower())).isalpha() and cw not in stopwords and len(cw) > 2
            # Filtra parole non alfabetiche, stopwords e parole corte
        ])
        wc = words.reduceByKey(lambda a, b: a + b)

        # --- 7) Top‑3 parole ---
        top3 = wc.map(lambda x: ((x[0][0], x[0][1], x[0][2]), (x[0][3], x[1]))) \
            .groupByKey() \
            .mapValues(lambda seq: [w for w, _ in sorted(seq, key=lambda x: -x[1])[:3]])

        # --- 8) Benchmark join e take(10) ---
        start = time.time()
        joined = grouped.join(top3)  # Unisci le statistiche numeriche con le top 3 parole
        sample = joined.take(10)  # Prendi un campione per visualizzazione
        duration = round(time.time() - start, 2)

        # --- 9) Output ---
        fout.write("Prime 10 risultati:\n")
        print("Prime 10 risultati:")
        for (c, y, b), ((cnt, avgd), topw) in sample:
            line = f"{c} | {y} | {b} | count={cnt} | avg_days={avgd} | top3={topw}"
            print(line)
            fout.write(line + "\n")

        fout.write(f"Tempo aggregation: {duration} sec\n")
        print(f"Tempo aggregation: {duration} sec")

        exec_times.append(duration)
        labels.append(label)

# 10) Grafico
plt.figure(figsize=(8, 5))
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

# Termina la SparkSession
spark.stop()