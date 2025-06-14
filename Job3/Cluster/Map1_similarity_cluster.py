#!/usr/bin/env python3

import sys
import csv
from io import StringIO

is_header = True
header_fields = []

for line in sys.stdin:
    if is_header:
        # Gestisce l'header del CSV
        header_fields = next(csv.reader(StringIO(line.strip())))
        is_header = False
        continue

    try:
        # Legge la riga come CSV per gestire virgole nei campi se presenti
        fields = next(csv.reader(StringIO(line.strip())))
        # Mappa i campi ai nomi di colonna
        row_dict = {header_fields[i]: fields[i] for i in range(len(fields))}

        # Pulizia e validazione dei dati
        model_name = str(row_dict.get('model_name', '')).strip()
        price_str = str(row_dict.get('price', '')).replace('$', '').replace(',', '').strip()
        horsepower_str = str(row_dict.get('horsepower', '')).strip()
        engine_displacement_str = str(row_dict.get('engine_displacement', '')).strip()

        price = float(price_str) if price_str else 0.0
        horsepower = float(horsepower_str) if horsepower_str else 0.0
        engine_displacement = float(engine_displacement_str) if engine_displacement_str else 0.0

        # Filtra le voci non valide (model_name vuoto o valori numerici non positivi)
        if (model_name and
                horsepower > 0 and
                engine_displacement > 0 and
                price > 0):
            # Output per il reducer: chiave=model_name, valore=somme parziali e contatore
            sys.stdout.write(f"{model_name}\t{horsepower},{engine_displacement},{price},{1}\n")

    except (ValueError, TypeError, KeyError) as e:
        # Scrive errori sullo stderr per debugging, ma salta la riga
        sys.stderr.write(f"Skipping malformed row in mapper_phase1: {line.strip()}. Error: {e}\n")
    except Exception as e:
        sys.stderr.write(f"Unexpected error in mapper_phase1 for row: {line.strip()}. Error: {e}\n")
