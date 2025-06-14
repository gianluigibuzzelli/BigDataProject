#!/usr/bin/env python3

import sys
from collections import defaultdict

if __name__ == "__main__":
    current_key = None
    # stats_agg: {key: [total_count, total_sum_days]}
    stats_agg = defaultdict(lambda: [0, 0.0])
    # word_counts_agg: {key: {word: count}}
    word_counts_agg = defaultdict(lambda: defaultdict(int))

    for line in sys.stdin:
        try:
            line = line.strip()
            if not line:  # Ignora righe vuote
                continue

            # La linea sarà nel formato "city,year,band\ttype,value1,value2..."
            key_str, value_str = line.split('\t', 1)

            # Parsa la chiave
            # Non è strettamente necessario ri-parsare i componenti della chiave qui
            # ma è buona pratica per chiarezza se volessi riutilizzarli.
            # city, year, band = key_str.split(',') # non usati direttamente qui ma utili per debug

            value_parts = value_str.split(',')
            value_type = value_parts[0]  # "stats" o "word"

            if value_type == "stats":
                count = int(value_parts[1])
                sum_days = float(value_parts[2])
                stats_agg[key_str][0] += count
                stats_agg[key_str][1] += sum_days
            elif value_type == "word":
                word = value_parts[1]
                word_count = int(value_parts[2])
                word_counts_agg[key_str][word] += word_count
            else:
                sys.stderr.write(f"Unknown value type: {value_type} in line: {line}\n")

        except Exception as e:
            sys.stderr.write(f"Error processing line in reducer: {line}. Error: {e}\n")
            continue

    # Dopo aver processato tutte le righe, calcoliamo e stampiamo i risultati finali
    # Ordina le chiavi per garantire un output consistente
    sorted_keys = sorted(stats_agg.keys())

    for key_str in sorted_keys:
        cnt, sum_days = stats_agg[key_str]

        # Calcola la media dei giorni sul mercato
        avg_days = round(sum_days / cnt, 2) if cnt > 0 else 0.0

        # Recupera le word counts per questa chiave
        wc = word_counts_agg.get(key_str, {})

        # Trova le top-3 parole (similmente al tuo codice originale)
        # Ordina le parole per conteggio (decrescente) e poi alfabeticamente (crescente)
        top3 = sorted(wc.items(), key=lambda item: (-item[1], item[0]))[:3]
        top3_words_list = [word for word, count in top3]  # Estrai solo le parole

        # Estrai i componenti della chiave per l'output leggibile
        city, year_str, band = key_str.split(',')
        year = int(year_str)  # Converti anno in int per l'output

        # Stampa il risultato finale nel formato desiderato
        sys.stdout.write(f"{city} | {year} | {band} | num={cnt} | avg_days={avg_days} | top3={top3_words_list}\n")
