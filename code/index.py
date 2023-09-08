import glob
import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from datetime import datetime


def create_month_dataFrame(my_path, no_test=True):
    datasets_files = glob.glob(my_path)
    # classement par ordre alphabetique des fichiers.
    datasets_files.sort()
    # création d'une liste pour stocker les DataFrames des fichiers dans le dossier
    df_list = []
    # itération sur la liste de fichiers csv dans le dossier
    for enum, file in enumerate(datasets_files):
        df_temp = pd.read_csv(file)
        if enum != 0:
            df_temp = df_temp.drop("category", axis=1)

        if "_up" in file:
            df_temp.columns = [col + "_up" for col in df_temp.columns]
        df_list.append(df_temp)

    # Concaténer horizontalement les DataFrames du dossier actuel
    df_month = pd.concat(df_list, axis=1, ignore_index=no_test)

    return df_month




def get_subdirectories(my_path):
    # Utilisez la fonction os.listdir() pour obtenir la liste de tous les fichiers et dossiers dans le chemin donné.
    # Ensuite, utilisez une liste en compréhension pour filtrer seulement les dossiers.
    subdirectories = [d for d in os.listdir(my_path) if os.path.isdir(os.path.join(my_path, d))]
    return subdirectories


parent = "../datasets"
df_annee_list = []

year_folders = get_subdirectories(parent)

for year_folder in year_folders:
    parent = "../datasets/" + year_folder
    month_folders = get_subdirectories(parent)
    for month_folder in month_folders:
        path = parent + "/" + month_folder + "/*.csv"

        df_annee_list.append(create_month_dataFrame(path))

df_test = create_month_dataFrame("../datasets/2022/janvier/*.csv", False)
df_test.to_csv("test.csv")
keys = list(range(269))
my_dict = dict(zip(keys, df_test.columns))

dataF = pd.concat(df_annee_list, axis=0, ignore_index=True)
dataF.rename(columns=my_dict, inplace=True)

dataF = dataF.dropna(axis=0)
dataF = pd.concat([dataF, (dataF['Minimum RTT'] != 0) & (dataF['Maximum RTT'] != 0)], axis=1)
dataF.rename(columns={0: 'test'}, inplace=True)
dataF.to_csv("dataF.csv")
print("Description du dataset")
print(dataF.describe())
print("Visualisation du dataset par la methode .head()")
print(dataF.head())
# download

# Latence

y_RTTVar_down = dataF['Maximum RTTVar']
dataF_features_RTTVar = [
    'Maximum RTO', 'Minimum RTT',
    'Maximum Backoff', 'Maximum RWndLimited', 'Maximum SndMSS',
    'Maximum RcvMSS', 'Maximum TotalRetrans', 'Maximum Reordering', 'Maximum Retransmits',
    'Maximum ATO', 'Minimum Retrans', 'Maximum BytesAcked', 'Average Lost', 'Maximum Sacked',
    'Maximum DataSegsOut', 'Maximum DataSegsIn',
    'Maximum SegsOut', 'Maximum SegsIn', 'Maximum BytesSent',
    'Maximum Download Bandwidth', 'Minimum Download Bandwidth', 'test'

]
X_latence_down = dataF[dataF_features_RTTVar]
train_X, val_X, train_y, val_y = train_test_split(X_latence_down, y_RTTVar_down, random_state=0, test_size=0.1,
                                                  train_size=0.9)
# Decision tree
print("Model arbre de décision")
bjnet_latence_down = DecisionTreeRegressor(random_state=1, max_leaf_nodes=500000)
bjnet_latence_down.fit(train_X, train_y)
val_predictions = bjnet_latence_down.predict(val_X)
print("Prédictions sur la latence du réseau : ")
print(val_predictions)
print("Taux d'erreur est de {}".format(mean_absolute_error(val_y, val_predictions)))

rttvar_preds = list(val_predictions)
rttvar_preds = [rtt for rtt in rttvar_preds if rtt != 0]
arr_rttvar_preds = np.array(rttvar_preds)
taux_precision_bonne = ((np.sum(arr_rttvar_preds < 100000.0)) / len(arr_rttvar_preds)) * 100
print("Le taux de performance du réseau (download) est de : {} %".format(taux_precision_bonne))
# indice de précision


# Random Forest
print("Model de Random Forest")
bjnet_latence_down = RandomForestRegressor(random_state=1)
bjnet_latence_down.fit(train_X, train_y)
val_predictions = bjnet_latence_down.predict(val_X)
print("prédiction sur la latence du réseau : ")
print(val_predictions)
print("Taux d'erreur est de {}".format(mean_absolute_error(val_y, val_predictions)))

rttvar_preds = list(val_predictions)
rttvar_preds = [rtt for rtt in rttvar_preds if rtt != 0]
arr_rttvar_preds = np.array(rttvar_preds)
taux_precision_bonne = ((np.sum(arr_rttvar_preds < 100000.0)) / len(arr_rttvar_preds)) * 100
print("Le taux de performance du réseau (download) est de : {} %".format(taux_precision_bonne))

# upload

y_RTTVar_up = dataF["Maximum RTTVar_up"]
dataF_features_RTTVar_up = [
    'test',
    'Maximum RTO_up', 'Minimum RTT_up',
    'Maximum Backoff_up', 'Maximum RWndLimited_up', 'Maximum SndMSS_up',
    'Maximum RcvMSS_up', 'Maximum TotalRetrans_up', 'Maximum Reordering_up', 'Maximum Retransmits_up',
    'Maximum ATO_up', 'Minimum Retrans_up', 'Maximum BytesAcked_up', 'Average Lost_up', 'Maximum Sacked_up',
    'Maximum DataSegsOut_up',
    'Maximum DataSegsIn_up',
    'Maximum SegsOut_up', 'Maximum SegsIn_up', 'Maximum BytesSent_up',
    'Maximum Upload Bandwidth', 'Minimum Upload Bandwidth',
]
X_latence_up = dataF[dataF_features_RTTVar_up]
X_train, X_val, y_train, y_val = train_test_split(X_latence_up, y_RTTVar_up, random_state=0, test_size=0.1,
                                                  train_size=0.9)
# Decision Tree
print("Modèle arbre de décision")
bjnet_latence_up = DecisionTreeRegressor(random_state=1, max_leaf_nodes=500000)
bjnet_latence_up.fit(X_train, y_train)
val_predictions = bjnet_latence_up.predict(X_val)
print("Prédictions sur la latence du réseau (upload) : ")
print(val_predictions)
print("Taux d'erreur est de {}".format(mean_absolute_error(val_y, val_predictions)))
rttvar_preds = list(val_predictions)
rttvar_preds = [rtt for rtt in rttvar_preds if rtt != 0]
arr_rttvar_preds = np.array(rttvar_preds)
taux_precision_bonne = ((np.sum(arr_rttvar_preds < 100000.0)) / len(arr_rttvar_preds)) * 100
print("Le taux de performance du réseau (upload) est de : {} %".format(taux_precision_bonne))
# Random forest
print("Modèle forêt random")
bjnet_latence_up = RandomForestRegressor(random_state=1, max_leaf_nodes=500000)
bjnet_latence_up.fit(X_train, y_train)
val_predictions = bjnet_latence_up.predict(X_val)
print("Prédiction sur la latence du réseau (upload) : ")
print(val_predictions)
print("Taux d'erreur est de {}".format(mean_absolute_error(val_y, val_predictions)))
rttvar_preds = list(val_predictions)

arr_rttvar_preds = np.array(rttvar_preds)
taux_precision_bonne = ((np.sum(arr_rttvar_preds < 100000.0)) / len(arr_rttvar_preds)) * 100
print("Le taux de performance du réseau (upload) est de : {} %".format(taux_precision_bonne))

# Visualisation et analyse des causes

# Analyse des paramêtres liés aux configurations TCP


# SndMSS, RcvMSS et PMTU

dataF_dec_2021 = pd.read_csv('./liste1.csv')
dates = [datetime.strptime(date, '%d %b %Y').date() for date in dataF_dec_2021['category']]
# download
# sndmss = dataF_dec_2021['Minimum SndMSS']
# rcvmss = dataF_dec_2021['Maximum RcvMSS']
# pmtu = dataF_dec_2021['Maximum PMTU']
# plt.plot(dates, pmtu, label="PMTU")
# plt.plot(dates, sndmss, label="SndMSS")
# plt.plot(dates, rcvmss, label="RcvMSS")
# plt.legend()
# plt.show()
# upload
# sndmss = dataF_dec_2021['Minimum SndMSS_up']
# rcvmss = dataF_dec_2021['Maximum RcvMSS_up']
# pmtu = dataF_dec_2021['Maximum PMTU_up']
# plt.plot(dates, pmtu, label='PMTU')
# plt.plot(dates, sndmss, label='SndMSS')
# plt.plot(dates, rcvmss, label='RcvMSS')
# plt.legend()
# plt.show()

# sacked & totalretrans
# download
sacked = dataF_dec_2021['Maximum Sacked']
# totalretrans = dataF_dec_2021['Maximum TotalRetrans']
# plt.plot(dates, sacked, label="Sacked")
# plt.plot(dates, totalretrans, label="TotalRetrans")
# plt.legend()
# plt.show()
# upload
# sacked = dataF_dec_2021['Maximum Sacked_up']
# totalretrans = dataF_dec_2021['Maximum TotalRetrans_up']
# plt.plot(dates, sacked, label="Sacked upload")
# plt.plot(dates, totalretrans, label="TotalRetrans upload")
# plt.legend()
# plt.show()

# RWndLimited & minRTT
# download
# rwndlimited = dataF_dec_2021['Maximum RWndLimited']
# minrtt = dataF_dec_2021['Minimum RTT']
# plt.plot(dates, rwndlimited, label="RwndLimited")
# plt.legend()
# plt.show()
#
# plt.plot(dates, minrtt, label="latence")
# plt.legend()
# plt.show()

# upload

# rwndlimited = dataF_dec_2021['Maximum RWndLimited_up']
# minrtt = dataF_dec_2021['Minimum RTT_up']
# plt.plot(dates, rwndlimited, label="RwndLimited upload")
# plt.legend()
# plt.show()
#
# plt.plot(dates, minrtt, label="latence upload")
# plt.legend()
# plt.show()
lost = dataF_dec_2021['Maximum Lost']
lost_up = dataF_dec_2021['Maximum Lost_up']
sacked_up = dataF_dec_2021['Maximum Sacked_up']
bytesRetrans = dataF_dec_2021['Maximum BytesRetrans']
bytesRetrans_up = dataF_dec_2021['Maximum BytesRetrans_up']
plt.plot(dates, lost, label="Quantité de segments perdus")
plt.legend()
plt.show()
plt.plot(dates, sacked, label="Quantité de segments marquées 'Sacked'")

plt.plot(dates, bytesRetrans, label="Quantité de données retransmises")
plt.legend()
plt.show()

#up
plt.plot(dates, lost_up, label="Quantité de segments perdus Upload")
plt.legend()
plt.show()
plt.plot(dates, sacked_up,label="Quantité de segments marquées 'Sacked' upload")


plt.plot(dates, bytesRetrans_up, label="Quantité de données retransmises Upload")
plt.legend()
plt.show()

"""
    Les visualisations des paramêtres seront faites avec les applications de visualisation de graphes
    Power Bi et Grafana. 
"""


