#!/usr/bin/env python3

import sys
import csv
import re
from io import StringIO

# Colonne essenziali
# Nota: 'price' non è tra le essential_cols perché viene processato come float
# e filtrato per range, non solo per presenza.
essential_cols = ['make_name', 'model_name', 'horsepower', 'engine_displacement', 'city', 'description']

# Espressioni regolari per la pulizia e l'estrazione
city_pattern = re.compile(r"^[A-Za-zÀ-ÖØ-öø-ÿ'\- ]{1,50}$")
word_pattern = re.compile(r"\b[a-zA-Z]{2,}\b")  # Trova solo parole di almeno 2 caratteri

# La prima riga dell'input è l'header. Deve essere saltata.

is_header = True
header_fields = []

for line in sys.stdin:
    if is_header:
        # Parsa l'header per ottenere i nomi delle colonne
        # StringIO permette a csv.reader di trattare la stringa come un file
        header_fields = next(csv.reader(StringIO(line.strip())))
        is_header = False
        continue  # Salta l'header, non lo processiamo come dato

    try:
        # Usa csv.reader per parsare la riga corrente
        fields = next(csv.reader(StringIO(line.strip())))

        # Crea un dizionario dalla riga, usando l'header per i nomi delle chiavi
        row_dict = {header_fields[i]: fields[i] for i in range(len(fields))}

        # --- Inizio Pulizia e Validazione Dati (replicata dal tuo codice originale) ---

        # Campi essenziali: verifica che non siano vuoti o 'NULL'
        for k in essential_cols:
            if not row_dict.get(k) or str(row_dict[k]).strip().upper() == 'NULL':
                raise ValueError(f"Campo essenziale mancante/nullo: {k}")

        # Price: pulizia e conversione
        price_str = str(row_dict.get('price', '0')).replace('$', '').replace(',', '').strip()
        price = float(price_str) if price_str else 0.0
        if not (0 < price < 1_000_000):
            raise ValueError(f"Prezzo non valido: {price}")

        # Year: conversione e validazione intervallo
        year_val = str(row_dict.get('year', '0')).strip()
        year = int(float(year_val)) if year_val and year_val.replace('.', '', 1).isdigit() else 0
        if not (1900 <= year <= 2025):
            raise ValueError(f"Anno non valido: {year}")

        # City: validazione con regex
        city = str(row_dict.get('city', '')).strip()
        if not city_pattern.match(city):
            raise ValueError(f"Città non valida: {city}")

        # DaysOnMarket: conversione
        days = int(float(row_dict.get('daysonmarket', '0'))) if str(row_dict.get('daysonmarket', '0')).strip().replace(
            '.', '', 1).isdigit() else 0

        # Description: pulizia e default a stringa vuota se None
        description = str(row_dict.get('description', '') or '').strip()

        # Determina la fascia di prezzo
        band = 'high' if price > 50000 else 'medium' if price >= 20000 else 'low'

        # --- Fine Pulizia e Validazione Dati ---

        # La chiave sarà (city, year, band)
        key_parts = [city, str(year), band]
        output_key = ",".join(key_parts)

        # Output per le statistiche (conteggio, somma giorni)
        # Formato: KEY\tTYPE_STATS,COUNT,SUM_DAYS
        sys.stdout.write(f"{output_key}\tstats,{1},{days}\n")

        # Output per le parole della descrizione
        # Formato: KEY\tTYPE_WORD,WORD,COUNT (count è sempre 1 in questa fase)
        for w in word_pattern.findall(description.lower()):
            sys.stdout.write(f"{output_key}\tword,{w},{1}\n")

    except (ValueError, TypeError, KeyError) as e:
        # Scrivi errori su stderr (vanno nei log di Hadoop)
        # Queste righe verranno scartate senza influenzare il job principale
        sys.stderr.write(f"Skipping malformed row: {line.strip()}. Error: {e}\n")
    except Exception as e:
        sys.stderr.write(f"Unexpected error in mapper for row: {line.strip()}. Error: {e}\n")
