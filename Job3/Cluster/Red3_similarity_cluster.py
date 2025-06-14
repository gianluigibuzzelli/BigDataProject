#!/usr/bin/env python3

import sys
import os
from collections import defaultdict


# Implementazione Union-Find per trovare i componenti connessi (i gruppi)
class UnionFind:
    def __init__(self, elements):
        # Inizializza ogni elemento come radice di se stesso
        self.parent = {e: e for e in elements}
        # Il rank (altezza dell'albero) aiuta a mantenere l'albero bilanciato durante l'unione
        self.rank = {e: 0 for e in elements}

    def find(self, i):
        # Trova la radice del componente di 'i' con compressione del percorso
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])  # Compressione del percorso
        return self.parent[i]

    def union(self, i, j):
        # Unisce i componenti di 'i' e 'j' per rank
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:  # Se sono in componenti diversi
            if self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
            elif self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
            else:
                self.parent[root_j] = root_i
                self.rank[root_i] += 1  # Incrementa il rank se sono uguali
            return True
        return False  # Già nello stesso componente


# all_model_features: Dizionario globale per memorizzare tutti i modelli con le loro features
# {model_name: (avg_hp, avg_ed, avg_price)}
all_model_features = {}


def load_all_model_features(cache_file_path):
    """Carica i dati di tutti i modelli dalla Distributed Cache (copia del mapper)."""
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


# Carica i dati dalla Distributed Cache all'avvio del reducer
if 'HADOOP_STREAMING_JAR' in os.environ:
    try:
        all_model_features = load_all_model_features('all_model_stats_data')
        sys.stderr.write(f"Reducer 3: Loaded {len(all_model_features)} models from distributed cache.\n")
    except Exception as e:
        sys.stderr.write(f"Reducer 3: Error loading distributed cache: {e}\n")

if __name__ == "__main__":
    # Questo reducer riceverà tutte le coppie (M1, M2) dove M1 e M2 sono simili.
    # Raccogli tutte le coppie di modelli simili
    all_similar_pairs = []
    unique_models_in_pairs = set()  # Per inizializzare Union-Find solo con i modelli pertinenti

    for line in sys.stdin:
        try:
            line = line.strip()
            if not line:
                continue

            # Input: model1 \t model2 (dal reducer_phase2)
            model1, model2 = line.split('\t', 1)
            all_similar_pairs.append((model1, model2))
            unique_models_in_pairs.add(model1)
            unique_models_in_pairs.add(model2)

        except Exception as e:
            sys.stderr.write(f"Error processing line in reducer_phase3 (collecting pairs): {line}. Error: {e}\n")
            continue

    if not unique_models_in_pairs:
        sys.stderr.write("Reducer 3: No similar pairs found. Exiting.\n")
        sys.exit(0)  # Nessuna coppia, niente da fare

    # Crea l'Union-Find su tutti i modelli che compaiono in almeno una coppia simile
    uf = UnionFind(list(unique_models_in_pairs))

    # Applica le unioni basate sulle coppie simili
    for m1, m2 in all_similar_pairs:
        uf.union(m1, m2)

    # Raggruppa i modelli per il loro "root" nella Union-Find, che rappresenta il gruppo connesso
    groups_by_root = defaultdict(list)  # {root_model: [list_of_members]}
    for model in unique_models_in_pairs:
        root = uf.find(model)
        groups_by_root[root].append(model)

    final_result = []

    # Per ogni gruppo trovato, calcola avg_price e top_power_model
    for group_root, members_list in sorted(groups_by_root.items()):
        # Rimuovi eventuali duplicati e ordina i membri per output consistente
        members = sorted(list(set(members_list)))

        total_price_for_group = 0.0
        total_members_with_valid_price = 0
        top_model_name_in_group = None
        max_hp_in_group = -1.0

        for member_model in members:
            # Recupera le features del membro dalla cache distribuita (all_model_features)
            hp, ed, price = all_model_features.get(member_model, (0.0, 0.0, 0.0))  # Default a 0 se non trovato

            if price > 0:  # Solo se ha un prezzo valido, contribuisce alla media
                total_price_for_group += price
                total_members_with_valid_price += 1

            # Trova il modello con la massima horsepower all'interno del gruppo
            if hp > max_hp_in_group:
                max_hp_in_group = hp
                top_model_name_in_group = member_model

        avg_price_for_group = round(total_price_for_group / total_members_with_valid_price,
                                    2) if total_members_with_valid_price > 0 else 0.0

        final_result.append((group_root, members, avg_price_for_group, top_model_name_in_group))

    # Ordina il risultato finale per il nome del gruppo radice per un output consistente
    final_result.sort(key=lambda x: x[0])

    # Stampa i risultati nel formato richiesto
    for grp_root, members, avgp, topm in final_result:
        sys.stdout.write(f"{grp_root} | members={members} | avg_price={avgp} | top_power_model={topm}\n")
