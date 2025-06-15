import csv
from io import StringIO
import re
import time
import os
import sys  

# Importa solo SparkSession, SparkContext è ottenuto da essa.
from pyspark.sql import SparkSession

# --- Configurazione per Cluster EMR ---

S3_CLEANED_INPUT_BASE_PATH = "s3://bucketpoggers2/input/"

# Il file di log e il file dei tempi verranno generati localmente sul nodo master EMR
# e poi scaricati.
LOCAL_LOG_DIR = "/home/hadoop/spark_logs_city_priceband_rdd/"  # Directory sul nodo master EMR per i log temporanei
LOCAL_LOG_FILE = os.path.join(LOCAL_LOG_DIR, "spark_core_city_priceband_rdd_only.txt")
LOCAL_TIMES_FILE_ON_EMR = os.path.join(LOCAL_LOG_DIR, "spark_core_city_priceband_rdd_only_times.txt")


os.makedirs(LOCAL_LOG_DIR, exist_ok=True)

# Inizializza SparkSession. Senza configurazioni esplicite.
# Spark userà le configurazioni di default del cluster EMR.
spark = SparkSession.builder \
    .appName("SparkCityPriceBandRDDCluster") \
    .getOrCreate()

# Imposta il livello di log per avere un output più pulito
spark.sparkContext.setLogLevel("ERROR")
sc = spark.sparkContext  # Ottieni lo SparkContext per le operazioni RDD

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
    ('used_cars_data_1_5x.csv', '150%'),  
    ('used_cars_data_2x.csv', '200%')
]

exec_times = []
labels = []

# stopwords facoltative (saranno serializzate e inviate ai workers)
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
        reader = csv.reader(StringIO(line))  # csv.reader in PySpark è robusto con default
        fields = next(reader)

        # Verifica che la riga abbia un numero sufficiente di campi per gli indici attesi
        # Questo cattura righe tronche prima di accedere agli indici
        min_required_idx = max(header_map.get('city', 0), header_map.get('year', 0),
                               header_map.get('price', 0), header_map.get('daysonmarket', 0),
                               header_map.get('description', 0))
        if len(fields) <= min_required_idx:
            return None  # Riga troppo corta

        # Estrazione e conversione dei campi necessari, con gestione degli errori
        city = fields[header_map['city']].strip()

        year = None
        year_str = fields[header_map['year']].strip()
        try:
            year = int(float(year_str))  # Gestisce "2024.0" -> 2024
        except ValueError:
            pass

        price = None
        price_str = fields[header_map['price']].strip()
        try:
            # Si assume che il file sia già pulito da valuta e spazi, ma per sicurezza...
            clean_price_str = re.sub(r'[^\d.]', '', price_str)
            price = float(clean_price_str)
        except ValueError:
            pass

        daysonmarket = None
        daysonmarket_str = fields[header_map['daysonmarket']].strip()
        try:
            daysonmarket = int(float(daysonmarket_str))
        except ValueError:
            pass

        description = fields[header_map['description']].strip() if 'description' in header_map else ""

        # Applicazione delle condizioni di filtro
        # Regex per la città: accetta lettere, spazi, apostrofi, trattini, virgole e punti.
        # Adatta questa regex se hai altri caratteri validi nei nomi delle città.
        city_valid = city and re.match(r"^[A-Za-zÀ-ÖØ-öø-ÿ0-9\s'\-,\.]{1,100}$", city)

        if not (price is not None and price > 0 and
                year is not None and 1900 <= year <= 2025 and
                city_valid):
            return None

            # Calcolo della banda di prezzo
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
            (city, year, band),  # Chiave: (city, year, band)
            (1, daysonmarket if daysonmarket is not None else 0, description)
        # Valore: (count, daysonmarket, description)
        )

    except (csv.Error, IndexError, StopIteration, ValueError) as e:
        # Cattura errori di parsing CSV, accesso a indici fuori bounds, righe vuote o conversioni
        # sys.stderr.write(f"Parsing error for line: {line.strip()}. Error: {e}\n") # Solo per debugging intenso
        return None
    except Exception as e:
        # Cattura qualsiasi altro errore imprevisto
        # sys.stderr.write(f"Unexpected error for line: {line.strip()}. Error: {e}\n") # Solo per debugging
        return None


with open(LOCAL_LOG_FILE, "w", encoding="utf-8") as fout:
    for file_name, label in datasets:
        s3_path = S3_CLEANED_INPUT_BASE_PATH + file_name
        fout.write(f"\n== Dataset {label}: {s3_path} ==\n")
        sys.stdout.write(f"\n== Dataset {label}: {s3_path} ==\n")

        try:
            # --- 3) Leggi CSV con Spark RDD ---
            raw_rdd = sc.textFile(s3_path)

            if raw_rdd.isEmpty():
                sys.stdout.write(f"AVVISO: Il file {s3_path} è vuoto o non accessibile. Salto questo dataset.\n\n")
                fout.write(f"AVVISO: Il file {s3_path} è vuoto o non accessibile. Salto questo dataset.\n\n")
                continue

            header_line = raw_rdd.first()
            data_rdd = raw_rdd.filter(lambda line: line != header_line)

            # Parsa l'header per creare una mappa nome_colonna -> indice
            # Questo è cruciale per accedere ai campi per nome (es. 'city') e non per indice fisso
            header_fields = next(csv.reader(StringIO(header_line)))
            header_map = {name.strip(): idx for idx, name in
                          enumerate(header_fields)}  # .strip() per pulire nomi header

            # Verifica che le colonne essenziali siano presenti
            required_cols = ['city', 'year', 'price', 'daysonmarket', 'description']
            if not all(col_name in header_map for col_name in required_cols):
                missing_cols = [col_name for col_name in required_cols if col_name not in header_map]
                sys.stdout.write(
                    f"ERRORE: Colonne essenziali mancanti nell'header di {s3_path}: {missing_cols}. Salto questo dataset.\n\n")
                fout.write(
                    f"ERRORE: Colonne essenziali mancanti nell'header di {s3_path}: {missing_cols}. Salto questo dataset.\n\n")
                continue

        except Exception as e:
            sys.stdout.write(
                f"ERRORE: Impossibile leggere o processare l'header del file {s3_path}. Salto questo dataset. Errore: {e}\n\n")
            fout.write(
                f"ERRORE: Impossibile leggere o processare l'header del file {s3_path}. Salto questo dataset. Errore: {e}\n\n")
            continue

        # Broadcast di header_map per renderlo disponibile efficientemente su tutti i worker
        header_map_broadcast = sc.broadcast(header_map)

        # --- 4) Applica parsing e filtraggio ---
        # Le righe malformate o non conformi ai filtri sono restituite come None e poi rimosse
        # Usiamo header_map_broadcast.value per accedere alla mappa broadcastata
        rdd_parsed = data_rdd.map(lambda line: parse_and_filter_csv_line(line, header_map_broadcast.value)) \
            .filter(lambda x: x is not None) \
            .cache()  # Cache per riutilizzo

        if rdd_parsed.isEmpty():
            sys.stdout.write(
                f"AVVISO: Nessun dato valido dopo il parsing e filtraggio per {s3_path}. Salto questo dataset.\n\n")
            fout.write(
                f"AVVISO: Nessun dato valido dopo il parsing e filtraggio per {s3_path}. Salto questo dataset.\n\n")
            # Rimuovi dalla cache broadcast
            header_map_broadcast.destroy()
            continue

        # --- 5) Statistiche numeriche (Spark Core) ---
        grouped_stats = rdd_parsed.mapValues(lambda v: (v[0], v[1])) \
            .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1])) \
            .mapValues(lambda t: (t[0], round(t[1] / t[0], 2) if t[0] > 0 else 0.0)) \
            .cache()  # Cache questo RDD intermedio

        # --- 6) Conteggio parole pulite ---
        words = rdd_parsed.flatMap(lambda x: [
            ((x[0][0], x[0][1], x[0][2], cw), 1)  # Key include city, year, band, cleaned_word
            for w in x[1][2].split()  # Access description from value tuple (x[1][2] è la description)
            if (cw := clean_word(w.lower())).isalpha() and cw not in stopwords and len(cw) > 2
        ])
        wc = words.reduceByKey(lambda a, b: a + b) \
            .cache()  # Cache il conteggio parole intermedio

        # --- 7) Top‑3 parole ---
        top3 = wc.map(lambda x: ((x[0][0], x[0][1], x[0][2]), (x[0][3], x[1]))) \
            .groupByKey() \
            .mapValues(
            lambda seq: [w for w, _ in sorted(list(seq), key=lambda x: -x[1])[:3]])  # sorted(list(seq)) per PySpark

        # --- 8) Benchmark join e take(10) ---
        start = time.time()
        joined = grouped_stats.join(top3)  # Unisci le statistiche numeriche con le top 3 parole
        sample = joined.take(10)  # Prendi un campione per visualizzazione
        duration = round(time.time() - start, 2)

        # --- 9) Output ---
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

        # Rilascia la cache dei RDD per liberare memoria
        rdd_parsed.unpersist()
        grouped_stats.unpersist()
        wc.unpersist()
        # Rimuovi dalla cache broadcast la mappa header
        header_map_broadcast.destroy()

# Scrivi i tempi di esecuzione in un file separato sul nodo master EMR
with open(LOCAL_TIMES_FILE_ON_EMR, "w", encoding="utf-8") as tf:
    for i in range(len(labels)):
        tf.write(f"{labels[i]} {exec_times[i]}\n")

# 10) Arresta SparkSession
spark.stop()

sys.stdout.write(f"\n✅ Report salvato in: {LOCAL_LOG_FILE}\n")
sys.stdout.write(f"✅ Tempi di esecuzione salvati in: {LOCAL_TIMES_FILE_ON_EMR}\n")
sys.stdout.write(f"Per generare il grafico, scarica '{LOCAL_TIMES_FILE_ON_EMR}' e usa uno script Python locale.\n")