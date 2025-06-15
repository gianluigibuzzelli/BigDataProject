#!/usr/bin/env python3

import sys
import os

# all_model_features: Dizionario globale per memorizzare tutti i modelli con le loro features
# {model_name: (avg_hp, avg_ed, avg_price)}
all_model_features = {}


def load_all_model_features(cache_file_path):
    """Carica i dati di tutti i modelli dalla Distributed Cache."""
    temp_model_features = {}
    try:
        with open(cache_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    model_name, values_str = line.split('\t', 1)
                    avg_hp, avg_ed, avg_price = map(float, values_str.split(','))
                    temp_model_features[model_name] = (avg_hp, avg_ed, avg_price)
                except ValueError as e:
                    sys.stderr.write(f"Error parsing cache line from {cache_file_path}: {line}. Error: {e}\n")
    except Exception as e:
        sys.stderr.write(f"Could not open or read Distributed Cache file {cache_file_path}. Error: {e}\n")
    return temp_model_features


# Carica i dati dalla Distributed Cache all'avvio del mapper
# 'all_model_stats_data' è il nome del symlink che Hadoop crea per il file nella cache
if 'HADOOP_STREAMING_JAR' in os.environ:
    # Hadoop symlinka i file aggiunti con -files/-cacheFile nella working directory del task.
    # all_model_stats_data è il nome che assegneremo con -cacheFile s3://...#all_model_stats_data
    try:
        all_model_features = load_all_model_features('all_model_stats_data')
        sys.stderr.write(f"Mapper 2: Loaded {len(all_model_features)} models from distributed cache.\n")
    except Exception as e:
        sys.stderr.write(f"Mapper 2: Error loading distributed cache: {e}\n")
 

# Il mapper riceve l'output della Fase 1 come input riga per riga.
# Per ogni modello di input, lo confronta con tutti i modelli dalla cache.
for line in sys.stdin:
    try:
        line = line.strip()
        if not line:
            continue

        # Parsa la riga di input (un singolo modello con le sue medie)
        current_model_name, current_values_str = line.split('\t', 1)
        current_hp, current_ed, current_price = map(float, current_values_str.split(','))

        # Confronta il modello corrente con tutti i modelli caricati dalla Distributed Cache.
        # Per assicurare che ogni coppia sia emessa una sola volta (M1,M2 vs M2,M1), ordiniamo i nomi.
        for other_model_name, (other_hp, other_ed, other_price) in all_model_features.items():
            if current_model_name == other_model_name:
                continue  # Salta il confronto con se stesso

            # Applica la condizione di similarità
            # Controlla la divisione per zero se hp o ed possono essere 0
            if current_hp > 0 and current_ed > 0:
                if (abs(current_hp - other_hp) / current_hp <= 0.10 and
                        abs(current_ed - other_ed) / current_ed <= 0.10):
                    # Trovata una coppia simile. Normalizza la chiave per il raggruppamento nel reducer.
                    # Emette sempre (modello_più_piccolo_alfabeticamente, modello_più_grande_alfabeticamente)
                    model_pair = tuple(sorted((current_model_name, other_model_name)))

                    # Output: chiave=model1,model2 (ordinati) \t valore=model1;model2 (modelli della coppia)
                    # Il valore è usato nel reducer per convalidare i membri della coppia
                    sys.stdout.write(f"{model_pair[0]},{model_pair[1]}\t{model_pair[0]};{model_pair[1]}\n")

    except Exception as e:
        sys.stderr.write(f"Unexpected error in mapper_phase2 for line: {line.strip()}. Error: {e}\n")
