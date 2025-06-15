#!/bin/bash

# --- Configurazione Generale ---
SSH_KEY_PATH="~/.ssh/pem/coppiadichiavi.pem" # Percorso alla tua chiave SSH
EMR_MASTER_HOST="ec2-18-206-12-179.compute-1.amazonaws.com" # <--- SOSTITUISCI CON IL TUO IP/DNS DEL MASTER EMR
SSH_USER="hadoop"

# Percorso dello script PySpark sul nodo master EMR
SPARK_SCRIPT_PATH="/home/${SSH_USER}/pog/cluster/spark_core_AVG_cluster.py"

# Directory dove i log e i tempi verranno temporaneamente salvati sul nodo master EMR
LOCAL_LOG_DIR_ON_EMR="/home/${SSH_USER}/spark_logs_simplified/"
LOCAL_TIMES_FILE_ON_EMR="${LOCAL_LOG_DIR_ON_EMR}spark_core_times.txt"
LOCAL_LOG_FILE_ON_EMR="${LOCAL_LOG_DIR_ON_EMR}spark_core_model_stats.txt"

# Directory locale dove scaricherai i risultati e genererai il grafico
LOCAL_RESULTS_DIR="/media/gianluigi/Z Slim/Risultati_cluster/"
LOCAL_GRAPH_FILE_PATH="${LOCAL_RESULTS_DIR}spark_core_model_stats_times.png"
LOCAL_TIMES_FILE_PATH="${LOCAL_RESULTS_DIR}spark_core_times.txt"

# Assicurati che la directory locale per i risultati esista
mkdir -p "${LOCAL_RESULTS_DIR}"

echo "Starting simplified Spark Core job on EMR cluster..."
echo "Results will be saved locally to: ${LOCAL_RESULTS_DIR}"

# Esegui lo script PySpark sul cluster EMR
ssh -i "${SSH_KEY_PATH}" "${SSH_USER}@${EMR_MASTER_HOST}" "\
  spark-submit \
  ${SPARK_SCRIPT_PATH} \
"

# Scarica i file di log e tempi dal nodo master EMR al tuo locale
echo "Downloading log and times files from EMR master to local machine..."
scp -i "${SSH_KEY_PATH}" "${SSH_USER}@${EMR_MASTER_HOST}:${LOCAL_LOG_FILE_ON_EMR}" "${LOCAL_RESULTS_DIR}" || true
scp -i "${SSH_KEY_PATH}" "${SSH_USER}@${EMR_MASTER_HOST}:${LOCAL_TIMES_FILE_ON_EMR}" "${LOCAL_RESULTS_DIR}" || true

echo "Spark job completed. Log and times files downloaded."

# Cleanup dei file temporanei sul nodo master EMR
ssh -i "${SSH_KEY_PATH}" "${SSH_USER}@${EMR_MASTER_HOST}" "rm -rf ${LOCAL_LOG_DIR_ON_EMR}" || true


# Ora, genera il grafico localmente (lo script Python è embeddato qui)
echo "Generating graph locally..."

python3 -c "
import matplotlib.pyplot as plt
import numpy as np
import os

output_dir = os.path.dirname('${LOCAL_GRAPH_FILE_PATH}')
graph_file = '${LOCAL_GRAPH_FILE_PATH}'
times_file = '${LOCAL_TIMES_FILE_PATH}'

exec_times = []
labels = []

try:
    with open(times_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    label, duration_str = line.split()
                    exec_times.append(float(duration_str))
                    labels.append(label)
                except ValueError:
                    print(f'Warning: Could not parse line in \'{times_file}\': {line}')
except FileNotFoundError:
    print(f'Error: Times file \'{times_file}\' not found. Please ensure the Spark job completed successfully and the file was downloaded.')
    exit(1)
except Exception as e:
    print(f'An error occurred while reading the times file: {e}')
    exit(1)

if not exec_times:
    print('No execution times found to plot. Exiting.')
    exit(0)

plt.figure(figsize=(10, 6))
plt.plot(labels, exec_times, marker='o', linestyle='-', color='b')
plt.title('Spark Core: Aggregation Time vs Dataset Size (EMR Cluster - Simplified Config)', fontsize=14)
plt.xlabel('Dataset Size (%)', fontsize=12)
plt.ylabel('Aggregation Time (s)', fontsize=12)
plt.ylim(bottom=0)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig(graph_file)
plt.close()

print(f'\n✅ Grafico dei tempi salvato in: {graph_file}')
"
