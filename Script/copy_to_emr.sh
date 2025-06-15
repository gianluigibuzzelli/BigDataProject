#!/bin/bash

# --- Configurazione ---
# Percorso della tua chiave SSH privata
SSH_KEY_PATH="~/.ssh/pem/coppiadichiavi.pem"

# Indirizzo IP/DNS del nodo master EMR
EMR_MASTER_HOST="ec2-44-200-144-233.compute-1.amazonaws.com"

# Utente SSH per EMR (solitamente 'hadoop')
SSH_USER="hadoop"

# Percorso della directory locale che vuoi copiare (contiene i tuoi script)
LOCAL_CLUSTER_PATH="/home/gianluigi/CodiciBigData/Progetto/OneMonth/cluster"

# Percorso della directory di destinazione sul nodo master EMR
REMOTE_DEST_PATH="/home/${SSH_USER}/pog"

# --- Inizio Operazioni ---

echo "Connessione a ${SSH_USER}@${EMR_MASTER_HOST} per creare la directory di destinazione..."

# Crea la directory di destinazione sul nodo master EMR
# L'opzione -p assicura che la directory venga creata solo se non esiste già
ssh -i "${SSH_KEY_PATH}" "${SSH_USER}@${EMR_MASTER_HOST}" "mkdir -p ${REMOTE_DEST_PATH}"

if [ $? -eq 0 ]; then
    echo "Directory ${REMOTE_DEST_PATH} creata o già esistente sul nodo master."
else
    echo "Errore durante la creazione della directory remota. Esco."
    exit 1
fi

echo "Copia dei file da ${LOCAL_CLUSTER_PATH} a ${REMOTE_DEST_PATH}..."

# Copia i file localmente (ricorsivamente) nella directory di destinazione remota
scp -i "${SSH_KEY_PATH}" -r "${LOCAL_CLUSTER_PATH}" "${SSH_USER}@${EMR_MASTER_HOST}:${REMOTE_DEST_PATH}/"

if [ $? -eq 0 ]; then
    echo "File copiati con successo."
else
    echo "Errore durante la copia dei file via SCP. Esco."
    exit 1
fi

echo "Impostazione dei permessi di esecuzione sui file Python remoti..."

# Imposta i permessi di esecuzione per gli script Python remoti
ssh -i "${SSH_KEY_PATH}" "${SSH_USER}@${EMR_MASTER_HOST}" "chmod +x ${REMOTE_DEST_PATH}/cluster/Map_AVG_cluster.py && chmod +x ${REMOTE_DEST_PATH}/cluster/Red_AVG_cluster.py"

if [ $? -eq 0 ]; then
    echo "Permessi di esecuzione impostati con successo."
else
    echo "Errore durante l'impostazione dei permessi. Verifica il percorso dei file remoti."
    exit 1
fi

echo "Operazione completata. I tuoi script sono ora in ${REMOTE_DEST_PATH}/cluster/ sul nodo master EMR."
