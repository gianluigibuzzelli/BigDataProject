import csv
import time
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict



datasets = [
    ('/media/gianluigi/Z Slim/DownloadUbuntu/archive/used_cars_data_sampled_1.csv', '10%'),
    ('/media/gianluigi/Z Slim/DownloadUbuntu/archive/used_cars_data_sampled_2.csv', '20%'),
    ('/media/gianluigi/Z Slim/DownloadUbuntu/archive/used_cars_data_sampled_3.csv', '30%'),
    ('/media/gianluigi/Z Slim/DownloadUbuntu/archive/used_cars_data_sampled_4.csv', '40%'),
    ('/media/gianluigi/Z Slim/DownloadUbuntu/archive/used_cars_data_sampled_5.csv', '50%'),
    ('/media/gianluigi/Z Slim/DownloadUbuntu/archive/used_cars_data_sampled_6.csv', '60%'),
    ('/media/gianluigi/Z Slim/DownloadUbuntu/archive/used_cars_data_sampled_7.csv', '70%'),
    ('/media/gianluigi/Z Slim/DownloadUbuntu/archive/used_cars_data_sampled_8.csv', '80%'),
    ('/media/gianluigi/Z Slim/DownloadUbuntu/archive/used_cars_data_sampled_9.csv', '90%'),
    ('/media/gianluigi/Z Slim/DownloadUbuntu/archive/used_cars_data.csv', '100%'),
    ('/media/gianluigi/Z Slim/DownloadUbuntu/archive/used_cars_data_2x.csv', '200%')
]


essential = ['make_name', 'model_name', 'horsepower', 'engine_displacement']
int_cols  = ['daysonmarket', 'dealer_zip', 'listing_id', 'owner_count', 'maximum_seating', 'year']

output_dir  = "/media/gianluigi/Z Slim/Risultati_locale/"
os.makedirs(output_dir, exist_ok=True)
log_file   = os.path.join(output_dir, "model_stats_mapreduce.txt")
graph_file = os.path.join(output_dir, "model_stats_mapreduce.png")

times = []
labels= []

# Funzione di pulizia di una riga
def clean_row(row):
    try:
        # Essenziali
        for k in essential:
            if not row.get(k) or row[k].upper()=='NULL':
                return None
        # Prezzo
        p = float(row['price'].replace('$','').replace(',','').strip())
        if p<=0 or p>=1_000_000: return None
        # Anno
        y = int(float(row['year']))
        if y<1900 or y>2025: return None
        # Cast interi
        for k in int_cols:
            v = row.get(k,'')
            row[k] = int(float(v)) if v and v.replace('.','',1).isdigit() else -1
        row['price']=p
        row['year']=y
        return row
    except:
        return None

with open(log_file,'w') as fout:
    for path,label in datasets:
        fout.write(f"\n== Dataset {label}: {path} ==\n")
        print(f"Processing {label}")

        # Aggregatori a chiave
        stats = {}  # (make,model)-> {count,min,max,sum, years_set}

        start = time.time()
        with open(path,newline='',encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                cr = clean_row(row)
                if cr is None: continue
                key=(cr['make_name'], cr['model_name'])
                if key not in stats:
                    stats[key] = [0, float('inf'), float('-inf'), 0.0, set()]
                cnt, mn, mx, sm, yrs = stats[key]
                cnt+=1; mn=min(mn, cr['price']); mx=max(mx, cr['price']); sm+=cr['price']; yrs.add(cr['year'])
                stats[key] = [cnt,mn,mx,sm,yrs]
       
        result=[]
        for (mk,md), (cnt,mn,mx,sm,yrs) in stats.items():
            result.append((mk,md,cnt,mn,mx,round(sm/cnt,2), sorted(yrs)))
        result.sort()
        duration = round(time.time() - start, 2)

        # Stampa prime 10
        for rec in result[:10]:
            line=f"{rec[0]} {rec[1]} | count={rec[2]} | min={rec[3]} | max={rec[4]} | avg={rec[5]} | years={rec[6]}"
            print(line)
            fout.write(line+"\n")

        print("Tempo MapReduce locale:",duration)
        fout.write(f"Tempo MapReduce locale: {duration} sec\n")
        times.append(duration)
        labels.append(label)

# Grafico tempi
plt.figure(figsize=(8,5))
plt.plot(labels,times,marker='o')
plt.title("MapReduce locale: Tempo vs Dimensione Dataset")
plt.xlabel("% dataset")
plt.ylabel("Sec")
plt.grid(True)
plt.tight_layout()
plt.savefig(graph_file)
plt.close()

print(f"Report e grafico salvati in {output_dir}")

