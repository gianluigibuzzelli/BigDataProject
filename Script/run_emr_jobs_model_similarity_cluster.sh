#!/bin/bash

# --- Configurazione Generale ---
SSH_KEY_PATH="~/.ssh/pem/coppiadichiavi.pem" # Percorso alla tua chiave SSH per EMR
EMR_MASTER_HOST="ec2-18-206-12-179.compute-1.amazonaws.com" # <--- SOSTITUISCI CON IL TUO IP/DNS DEL MASTER EMR
SSH_USER="hadoop" # Utente SSH di default per EMR

# Percorsi S3 per input/output
S3_INPUT_BASE_PATH="s3://bucketpoggers2/input/"
S3_PHASE1_OUTPUT_BASE_PATH="s3://bucketpoggers2/intermediate/phase1_model_stats/" # Output Fase 1
S3_PHASE2_OUTPUT_BASE_PATH="s3://bucketpoggers2/intermediate/phase2_similar_pairs/" # Output Fase 2
S3_FINAL_OUTPUT_BASE_PATH="s3://bucketpoggers2/output/model_similarity_results/" # Output Finale Fase 3

# Percorsi degli script Python sul nodo master EMR
# Assicurati che questi percorsi corrispondano a dove li hai copiati.
MAPPER_PHASE1_SCRIPT_PATH="/home/${SSH_USER}/pog/cluster/Map1_similarity_cluster.py"
REDUCER_PHASE1_SCRIPT_PATH="/home/${SSH_USER}/pog/cluster/Red1_similarity_cluster.py"
MAPPER_PHASE2_SCRIPT_PATH="/home/${SSH_USER}/pog/cluster/Map2_similarity_cluste.py"
REDUCER_PHASE2_SCRIPT_PATH="/home/${SSH_USER}/pog/cluster/Red2_similarity_cluster.py"
REDUCER_PHASE3_SCRIPT_PATH="/home/${SSH_USER}/pog/cluster/Red3_similarity_cluster.py"

NUM_REDUCE_TASKS_NORMAL=15
NUM_REDUCE_TASKS_PHASE3=1 

# Definizione dei dataset (nomi file S3 e etichette per il grafico)
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
    "used_cars_data_1_5x.csv 150%"
    "used_cars_data_2x.csv 200%"
)

# File locale dove verranno salvati i tempi di esecuzione per il grafico
LOCAL_TIMES_FILE="emr_model_similarity_cluster_times.txt"

# Inizializza il file dei tempi
echo "Inizio benchmark Hadoop Streaming su EMR (Model Similarity - 3 Fasi)" > "${LOCAL_TIMES_FILE}"
echo "------------------------------------------------------------------" >> "${LOCAL_TIMES_FILE}"

# Loop attraverso ogni dataset per eseguire le 3 fasi
for dataset_entry in "${DATASETS[@]}"; do
    read -r S3_FILE_NAME LABEL <<< "${dataset_entry}"

    S3_INPUT_PATH="${S3_INPUT_BASE_PATH}${S3_FILE_NAME}"
    # Percorsi di output specifici per il dataset e la fase
    S3_PHASE1_OUTPUT_DIR="${S3_PHASE1_OUTPUT_BASE_PATH}$(echo "${LABEL}" | tr -d '%')"
    S3_PHASE2_OUTPUT_DIR="${S3_PHASE2_OUTPUT_BASE_PATH}$(echo "${LABEL}" | tr -d '%')"
    S3_FINAL_OUTPUT_DIR="${S3_FINAL_OUTPUT_BASE_PATH}$(echo "${LABEL}" | tr -d '%')"
    
    echo ""
    echo "--- Elaborazione Dataset: ${LABEL} (${S3_FILE_NAME}) ---"
    TOTAL_START_TIME=$(date +%s) # Inizio misurazione tempo totale per il dataset

    # --- FASE 1: Calcolo Medie Modello ---
    echo "Starting Phase 1: Calculating Model Averages..."
    echo "  Input: ${S3_INPUT_PATH}"
    echo "  Output: ${S3_PHASE1_OUTPUT_DIR}"
    # Pulisci la directory di output precedente
    ssh -i "${SSH_KEY_PATH}" "${SSH_USER}@${EMR_MASTER_HOST}" "hadoop fs -rm -r -skipTrash ${S3_PHASE1_OUTPUT_DIR} || true"

    HADOOP_CMD_PHASE1="hadoop jar /usr/lib/hadoop-mapreduce/hadoop-streaming.jar \
    -D mapreduce.job.reduces=${NUM_REDUCE_TASKS_NORMAL} \
    -input ${S3_INPUT_PATH} \
    -output ${S3_PHASE1_OUTPUT_DIR} \
    -mapper ${MAPPER_PHASE1_SCRIPT_PATH} \
    -reducer ${REDUCER_PHASE1_SCRIPT_PATH} \
    -file ${MAPPER_PHASE1_SCRIPT_PATH} \
    -file ${REDUCER_PHASE1_SCRIPT_PATH}"
    
    PHASE1_START_TIME=$(date +%s)
    ssh -i "${SSH_KEY_PATH}" "${SSH_USER}@${EMR_MASTER_HOST}" "${HADOOP_CMD_PHASE1}" &> "hadoop_phase1_output_${LABEL}.log"
    PHASE1_DURATION=$(( $(date +%s) - PHASE1_START_TIME ))
    echo "Phase 1 completed in ${PHASE1_DURATION} seconds."

    # --- Preparazione per Fase 2: Unire output Fase 1 per la Distributed Cache ---

    TEMP_MODEL_STATS_FILE="/home/${SSH_USER}/pog/cluster/all_model_stats_${LABEL}.txt"
    echo "Preparing Distributed Cache for Phase 2/3: Merging Phase 1 output to ${TEMP_MODEL_STATS_FILE}..."
    ssh -i "${SSH_KEY_PATH}" "${SSH_USER}@${EMR_MASTER_HOST}" "hadoop fs -getmerge ${S3_PHASE1_OUTPUT_DIR} ${TEMP_MODEL_STATS_FILE} || true"
    
    # --- FASE 2: Generazione Coppie Simili ---
    echo "Starting Phase 2: Generating Similar Pairs..."
    echo "  Input: ${S3_PHASE1_OUTPUT_DIR}" # L'input formale è l'output della Fase 1
    echo "  Output: ${S3_PHASE2_OUTPUT_DIR}"
    ssh -i "${SSH_KEY_PATH}" "${SSH_USER}@${EMR_MASTER_HOST}" "hadoop fs -rm -r -skipTrash ${S3_PHASE2_OUTPUT_DIR} || true"

    # Aggiungi il file unito (`TEMP_MODEL_STATS_FILE`) alla Distributed Cache.

    HADOOP_CMD_PHASE2="hadoop jar /usr/lib/hadoop-mapreduce/hadoop-streaming.jar \
    -D mapreduce.job.reduces=${NUM_REDUCE_TASKS_NORMAL} \
    -input ${S3_PHASE1_OUTPUT_DIR} \
    -output ${S3_PHASE2_OUTPUT_DIR} \
    -mapper ${MAPPER_PHASE2_SCRIPT_PATH} \
    -reducer ${REDUCER_PHASE2_SCRIPT_PATH} \
    -file ${MAPPER_PHASE2_SCRIPT_PATH} \
    -file ${REDUCER_PHASE2_SCRIPT_PATH} \
    -file ${TEMP_MODEL_STATS_FILE}#all_model_stats_data" # File della cache con symlink
    
    PHASE2_START_TIME=$(date +%s)
    ssh -i "${SSH_KEY_PATH}" "${SSH_USER}@${EMR_MASTER_HOST}" "${HADOOP_CMD_PHASE2}" &> "hadoop_phase2_output_${LABEL}.log"
    PHASE2_DURATION=$(( $(date +%s) - PHASE2_START_TIME ))
    echo "Phase 2 completed in ${PHASE2_DURATION} seconds."

    # --- FASE 3: Calcolo Componenti Connessi e Statistiche Finali ---
    echo "Starting Phase 3: Calculating Connected Components and Final Statistics..."
    echo "  Input: ${S3_PHASE2_OUTPUT_DIR}"
    echo "  Output: ${S3_FINAL_OUTPUT_DIR}"
    ssh -i "${SSH_KEY_PATH}" "${SSH_USER}@${EMR_MASTER_HOST}" "hadoop fs -rm -r -skipTrash ${S3_FINAL_OUTPUT_DIR} || true"

    # La Fase 3 usa UN SOLO REDUCER per garantire che tutti i dati dei bordi del grafo

    HADOOP_CMD_PHASE3="hadoop jar /usr/lib/hadoop-mapreduce/hadoop-streaming.jar \
    -D mapreduce.job.reduces=${NUM_REDUCE_TASKS_PHASE3} \
    -input ${S3_PHASE2_OUTPUT_DIR} \
    -output ${S3_FINAL_OUTPUT_DIR} \
    -mapper /usr/bin/cat \
    -reducer ${REDUCER_PHASE3_SCRIPT_PATH} \
    -file ${REDUCER_PHASE3_SCRIPT_PATH} \
    -file ${TEMP_MODEL_STATS_FILE}#all_model_stats_data" # File della cache con symlink
    
    PHASE3_START_TIME=$(date +%s)
    ssh -i "${SSH_KEY_PATH}" "${SSH_USER}@${EMR_MASTER_HOST}" "${HADOOP_CMD_PHASE3}" &> "hadoop_phase3_output_${LABEL}.log"
    PHASE3_DURATION=$(( $(date +%s) - PHASE3_START_TIME ))
    echo "Phase 3 completed in ${PHASE3_DURATION} seconds."

    # Pulizia del file temporaneo della cache sul nodo master
    ssh -i "${SSH_KEY_PATH}" "${SSH_USER}@${EMR_MASTER_HOST}" "rm ${TEMP_MODEL_STATS_FILE} || true"

    TOTAL_DURATION=$(( $(date +%s) - TOTAL_START_TIME ))
    echo "Total process for ${LABEL} completed in ${TOTAL_DURATION} seconds."
    echo "${LABEL} ${TOTAL_DURATION}" >> "${LOCAL_TIMES_FILE}"

done

echo ""
echo "--------------------------------------"
echo "All Hadoop Streaming jobs on EMR (Model Similarity) completed."
echo "Total execution times are saved in ${LOCAL_TIMES_FILE}"

echo "Now, you can use a local Python script (e.g., 'generate_graph.py') to plot these times."
