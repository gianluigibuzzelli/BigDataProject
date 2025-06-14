#!/usr/bin/env python3

import sys
import collections  # Per defaultdict, utile se vuoi aggregare in locale prima di stampare
import re


# Funzione per aggregare le statistiche
def aggregate_stats(current_stats, new_value_parts):
    """
    Combina le statistiche accumulate con i nuovi valori.
    current_stats: (count, min_price, max_price, sum_price, set_of_years)
    new_value_parts: (1, price, price, price, year)
    """
    new_count = current_stats[0] + new_value_parts[0]
    new_min = min(current_stats[1], new_value_parts[1])
    new_max = max(current_stats[2], new_value_parts[2])
    new_sum = current_stats[3] + new_value_parts[3]
    # Aggiungi il nuovo anno al set
    new_years = current_stats[4].union({new_value_parts[4]})  # new_value_parts[4] è l'anno

    return (new_count, new_min, new_max, new_sum, new_years)


if __name__ == "__main__":
    current_key = None
    # Inizializza le statistiche accumulate per la chiave corrente:
    # (count, min_price, max_price, sum_price, set_of_years)
    accumulated_stats = (0, float('inf'), float('-inf'), 0.0, set())

    for line in sys.stdin:
        try:
            # Dividi la linea in chiave e valore usando il tab
            key_str, value_str = line.strip().split('\t', 1)

            # Parsa la chiave (make_name, model_name)
            make_name, model_name = key_str.split(',')

            # Parsa il valore (count, price, price, price, year)
            # Usa float per i prezzi, int per count e year
            parts = [float(p) for p in value_str.split(',')]
            current_value_tuple = (int(parts[0]), parts[1], parts[2], parts[3], int(parts[4]))

            # Se la chiave è cambiata, elabora e stampa i risultati della chiave precedente
            if current_key and current_key != (make_name, model_name):
                # Calcola la media finale
                avg_price = round(accumulated_stats[3] / accumulated_stats[0], 2)
                # Ordina gli anni
                sorted_years = sorted(list(accumulated_stats[4]))

                # Stampa il risultato per la chiave precedente
                sys.stdout.write(
                    f"{current_key[0]} {current_key[1]} | count={accumulated_stats[0]} | min={accumulated_stats[1]} | max={accumulated_stats[2]} | avg={avg_price} | years={sorted_years}\n")

                # Resetta per la nuova chiave
                accumulated_stats = (0, float('inf'), float('-inf'), 0.0, set())

            current_key = (make_name, model_name)

            # Aggrega i valori
            accumulated_stats = aggregate_stats(accumulated_stats, current_value_tuple)

        except Exception as e:
            sys.stderr.write(f"Errore nel reducer sulla riga: {line.strip()}. Errore: {e}\n")

    # Non dimenticare di elaborare e stampare l'ultima chiave dopo il ciclo
    if current_key:
        avg_price = round(accumulated_stats[3] / accumulated_stats[0], 2)
        sorted_years = sorted(list(accumulated_stats[4]))
        sys.stdout.write(
            f"{current_key[0]} {current_key[1]} | count={accumulated_stats[0]} | min={accumulated_stats[1]} | max={accumulated_stats[2]} | avg={avg_price} | years={sorted_years}\n")
