#!/usr/bin/env python3

import sys
from collections import defaultdict

if __name__ == "__main__":
    # defaultdict per accumulare le somme e i conteggi per ogni modello
    # [sum_hp, sum_ed, sum_price, count]
    model_data = defaultdict(lambda: [0.0, 0.0, 0.0, 0])

    for line in sys.stdin:
        try:
            line = line.strip()
            if not line: # Salta righe vuote
                continue

            # Parsing input: model_name \t hp,ed,price,count (count sarà sempre 1 dal mapper)
            model_name, values_str = line.split('\t', 1)
            hp, ed, price, count = map(float, values_str.split(','))

            # Aggrega i valori per il modello corrente
            model_data[model_name][0] += hp
            model_data[model_name][1] += ed
            model_data[model_name][2] += price
            model_data[model_name][3] += count

        except Exception as e:
            sys.stderr.write(f"Error processing line in reducer_phase1: {line}. Error: {e}\n")
            continue

    # Dopo aver elaborato tutti i dati (il reducer riceve tutti i valori per ogni chiave ordinati),
    # calcola le medie finali e le emette.
    # Ordina i modelli per output consistente (opzionale, ma buona pratica)
    for model_name, (hp_sum, ed_sum, price_sum, count) in sorted(model_data.items()):
        if count > 0: # Evita divisione per zero
            avg_hp = round(hp_sum / count, 2)
            avg_ed = round(ed_sum / count, 2)
            avg_price = round(price_sum / count, 2)
            # Output: model_name \t avg_hp,avg_ed,avg_price
            sys.stdout.write(f"{model_name}\t{avg_hp},{avg_ed},{avg_price}\n")
