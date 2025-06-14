#!/usr/bin/env python3

import sys
import csv
import re
from io import StringIO

# Definizioni delle colonne essenziali e intere
essential_cols = ['make_name', 'model_name', 'horsepower', 'engine_displacement']
int_cols = ['daysonmarket', 'dealer_zip', 'listing_id', 'owner_count', 'maximum_seating', 'year']


# Funzione per pulire e validare una riga di dati
def clean_row(row_dict):
    """
    Pulisce e valida una riga di dati, gestendo i tipi e i valori nulli.
    Riceve un dizionario `row_dict` che rappresenta una riga del CSV.
    """
    try:
        current_row = row_dict

        # Essenziali: verifica che i campi non siano vuoti o 'NULL'
        for k in essential_cols:
            if not current_row.get(k) or str(current_row[k]).strip().upper() == 'NULL':
                return None

        # Prezzo: pulizia e conversione
        p_str = str(current_row.get('price', '')).replace('$', '').replace(',', '').strip()
        p = float(p_str)
        if p <= 0 or p >= 1_000_000:
            return None

        # Anno: conversione e validazione intervallo
        y_str = str(current_row.get('year', '')).strip()
        y = int(float(y_str))  # Gestisce "2024.0"
        if y < 1900 or y > 2025:
            return None

        # Cast interi: conversione e gestione valori non numerici
        for k in int_cols:
            v_str = str(current_row.get(k, '')).strip()
            if v_str and re.match(r'^-?\d+(\.\d+)?$', v_str):  # Controlla se è un numero valido (int o float)
                current_row[k] = int(float(v_str))
            else:
                current_row[k] = -1  # Valore di default per campi non validi

        current_row['price'] = p
        current_row['year'] = y
        return current_row
    except (ValueError, KeyError, TypeError):
        # Cattura errori di conversione o chiavi mancanti
        return None
    except Exception as e:
        # Cattura qualsiasi altro errore imprevisto
        # sys.stderr.write(f"Errore in clean_row: {e}, riga: {row_dict}\n")
        return None


def get_header(input_stream):
    """
    Legge la prima riga dallo stream e la usa come header.
    Ritorna la lista dei campi dell'header e riposiziona lo stream per la prossima lettura.
    Questa funzione non è usata direttamente nel mapper streaming ma serve per logica.
    In Hadoop Streaming, l'header va gestito escludendolo dall'input o scartandolo nel mapper.
    """
    # N.B. In Hadoop Streaming, non puoi "riposizionare" stdin.
    # Il mapper riceve le righe una per una. L'header deve essere saltato.
    pass


if __name__ == "__main__":
    # La prima riga dell'input è l'header. Deve essere saltata.
    # Se il tuo input è garantito essere senza header, puoi rimuovere questa parte.
    # sys.stdin è uno stream di righe.
    is_header = True
    header_fields = []

    for line in sys.stdin:
        if is_header:
            # Parsa l'header per ottenere i nomi delle colonne
            header_fields = next(csv.reader(StringIO(line.strip())))
            is_header = False
            continue  # Salta l'header, non lo processiamo come dato

        try:
            # Usa csv.reader per parsare la riga corrente
            # StringIO permette di trattare la stringa come un file per csv.reader
            fields = next(csv.reader(StringIO(line.strip())))

            # Crea un dizionario dalla riga, usando l'header per i nomi delle chiavi
            row_dict = {header_fields[i]: fields[i] for i in range(len(fields))}

            # Pulisci e valida la riga
            cleaned_row = clean_row(row_dict)

            if cleaned_row:
                # Chiave: (make_name, model_name)
                key = (cleaned_row['make_name'], cleaned_row['model_name'])
                # Valore: (count, min_price, max_price, sum_price, year)
                # In questa fase, min, max, sum sono tutti il prezzo corrente.
                # L'anno viene aggiunto come singolo elemento per il set nella fase di reduce.
                value = (1, cleaned_row['price'], cleaned_row['price'], cleaned_row['price'], cleaned_row['year'])

                # Stampa la coppia chiave-valore nel formato richiesto da Hadoop Streaming:
                # key_part1,key_part2\tval1,val2,val3,val4,val5
                # Usiamo la virgola per i campi interni e il tab come separatore chiave-valore
                sys.stdout.write(f"{key[0]},{key[1]}\t{value[0]},{value[1]},{value[2]},{value[3]},{value[4]}\n")
        except Exception as e:
            # Scrivi errori su stderr (vanno nei log di Hadoop)
            sys.stderr.write(f"Errore nel mapper sulla riga: {line.strip()}. Errore: {e}\n")