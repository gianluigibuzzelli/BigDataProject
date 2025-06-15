#!/bin/bash

# --- Configurazione ---
SSH_KEY_PATH="~/.ssh/pem/coppiadichiavi.pem"
EMR_MASTER_HOST="ec2-44-200-144-233.compute-1.amazonaws.com" # <--- SOSTITUISCI CON IL TUO IP/DNS DEL MASTER EMR
SSH_USER="hadoop"

S3_INPUT_BASE_PATH="s3://bucketpoggers2/input/"
S3_OUTPUT_BASE_PATH="s3://bucketpoggers2/output/"

MAPPER_SCRIPT_PATH="/home/${SSH_USER}/pog/cluster/Map_AVG_cluster.py"
REDUCER_SCRIPT_PATH="/home/${SSH_USER}/pog/cluster/Red_AVG_cluster.py"

NUM_REDUCE_TASKS=15 # Numero di reducer per il cluster di 5 nodi

# Definizione dei dataset: (nome_file_s3, label_percentuale)

DATASETS=(
    "used_cars_data_sampled_1.csv 10%"
    "used_cars_data_sampled_2.csv 20%"
    "used_cars_data_sampled_3.csv 30%"
    "used_cars_data_sampled_4.csv 40%"
    "used_cars_data_sampled_5.csv 50%"
    "used_cars_data_sampled_6.csv 60%"
    "used_cars_data_sampled_7.csv 70%"
    "used_cars_data_sampled_8.csv 80%"
    "used_cars_data_sampled_9.csv 90%"
    "used_cars_data.csv 100%"
    "used_cars_data_1_25x.csv 125%"
    "used_cars_data_1_5x.csv 150%"
    "used_cars_data_2x.csv 200%"
)

# File locale dove verranno salvati i tempi di esecuzione
LOCAL_TIMES_FILE="emr_job_times.txt"

# Inizializza il file dei tempi
echo "Inizio benchmark Hadoop Streaming su EMR" > "${LOCAL_TIMES_FILE}"
echo "--------------------------------------" >> "${LOCAL_TIMES_FILE}"

# Loop attraverso ogni dataset
for dataset_entry in "${DATASETS[@]}"; do
    # Estrai nome file e label
    read -r S3_FILE_NAME LABEL <<< "${dataset_entry}"

    S3_INPUT_PATH="${S3_INPUT_BASE_PATH}${S3_FILE_NAME}"
    S3_OUTPUT_DIR="${S3_OUTPUT_BASE_PATH}results_$(echo "${LABEL}" | tr -d '%')" # es: results_10
    
    echo ""
    echo "--- Elaborazione Dataset: ${LABEL} (${S3_FILE_NAME}) ---"
    echo "Input S3: ${S3_INPUT_PATH}"
    echo "Output S3: ${S3_OUTPUT_DIR}"

    # Pulisci la directory di output S3 prima dell'esecuzione
    echo "Pulizia della directory di output S3: ${S3_OUTPUT_DIR}..."
    ssh -i "${SSH_KEY_PATH}" "${SSH_USER}@${EMR_MASTER_HOST}" "hadoop fs -rm -r -skipTrash ${S3_OUTPUT_DIR} || true"
    # '|| true' per ignorare errori se la directory non esiste

    # Costruisci il comando Hadoop Streaming
    HADOOP_COMMAND="hadoop jar /usr/lib/hadoop-mapreduce/hadoop-streaming.jar \
    -input ${S3_INPUT_PATH} \
    -output ${S3_OUTPUT_DIR} \
    -mapper ${MAPPER_SCRIPT_PATH} \
    -reducer ${REDUCER_SCRIPT_PATH} \
    -file ${MAPPER_SCRIPT_PATH} \
    -file ${REDUCER_SCRIPT_PATH} \
    -numReduceTasks ${NUM_REDUCE_TASKS}"

    echo "Esecuzione del job Hadoop Streaming..."
    START_TIME=$(date +%s) # Tempo di inizio in secondi

    # Esegui il comando Hadoop Streaming sul nodo master EMR
    ssh -i "${SSH_KEY_PATH}" "${SSH_USER}@${EMR_MASTER_HOST}" "${HADOOP_COMMAND}" &> "hadoop_job_output_${LABEL}.log"

    END_TIME=$(date +%s) # Tempo di fine in secondi
    DURATION=$((END_TIME - START_TIME))

    echo "Job completato per ${LABEL} in ${DURATION} secondi."
    echo "${LABEL} ${DURATION}" >> "${LOCAL_TIMES_FILE}"

done

echo ""
echo "--------------------------------------"
echo "Tutti i job Hadoop Streaming completati."
echo "I tempi di esecuzione sono salvati in ${LOCAL_TIMES_FILE}"
echo "Ora puoi eseguire lo script Python per plottare il grafico."

# Il resto dello script verrà eseguito dopo il completamento di tutti i job Hadoop.
# Per ora, questo script si ferma qui e ti chiederà di eseguire plot_results.py manualmente.
