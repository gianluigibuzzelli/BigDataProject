#!/usr/bin/env python3

import sys

# Questo reducer riceve coppie ordinate (M1,M2) e le emette per la fase successiva.
# L'obiettivo è eliminare eventuali duplicati di emissione dal mapper (se un mapper A emette (X,Y) e un mapper B emette (X,Y))
# e preparare i dati per il calcolo dei componenti connessi.

# La chiave è la coppia M1,M2 (ordinata alfabeticamente).
# Il valore è M1;M2 (i modelli della coppia).
# Hadoop raggrupperà tutte le occorrenze della stessa chiave (M1,M2) e le passerà qui.

# current_key_pair: memorizza la coppia di modelli che stiamo processando
current_key_pair = None

for line in sys.stdin:
    try:
        line = line.strip()
        if not line:
            continue

        # Input: model1,model2 \t model1;model2
        key_pair_str, models_str = line.split('\t', 1)

        # Poiché Hadoop aggrega le righe con la stessa chiave, questo reducer
        # riceverà tutte le occorrenze di una specifica coppia (model1,model2).
        # Vogliamo emettere questa coppia solo una volta per il prossimo job.

        if key_pair_str != current_key_pair:
            # Se la chiave è cambiata, significa che stiamo processando una nuova coppia unica.
            # Emettiamo la coppia una sola volta.
            # Output per la fase 3: model1 \t model2 (rappresenta un bordo nel grafo)
            m1, m2 = key_pair_str.split(',')
            sys.stdout.write(f"{m1}\t{m2}\n")
            # Emesso anche il bordo inverso per facilitare la ricerca dei componenti connessi
            sys.stdout.write(f"{m2}\t{m1}\n")
            current_key_pair = key_pair_str  # Aggiorna la chiave corrente

    except Exception as e:
        sys.stderr.write(f"Error processing line in reducer_phase2: {line}. Error: {e}\n")
        continue
