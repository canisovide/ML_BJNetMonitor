import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Données pour les dates au format '1 Feb 2020'
dates_str = ['1 Feb 2020', '5 Feb 2020', '10 Feb 2020', '15 Feb 2020', '20 Feb 2020',
             '1 Mar 2020', '5 Mar 2020', '10 Mar 2020', '15 Mar 2020', '20 Mar 2020',
             '1 Sep 2021', '5 Sep 2021', '10 Sep 2021', '15 Sep 2021', '20 Sep 2021']

# Convertir les dates en format de date Python
dates = [datetime.strptime(date, '%d %b %Y').date() for date in dates_str]

# Données pour sndMSS, rcvMSS et PMTU
sndMSS = [1400, 1500, 1600, 1550, 1650, 1600, 1700, 1750, 1800, 1820, 1600, 1620, 1650, 1700, 1750]
rcvMSS = [1350, 1450, 1550, 1600, 1500, 1500, 1600, 1650, 1700, 1720, 1500, 1520, 1550, 1600, 1650]
PMTU = [1400, 1420, 1500, 1550, 1600, 1500, 1520, 1600, 1650, 1700, 1450, 1480, 1500, 1550, 1600]

# Créer le graphique initial avec les courbes de septembre 2021 visibles
fig, ax = plt.subplots(figsize=(10, 6))
lines = []
visible = {}

for i in range(len(dates_str)):
    line, = ax.plot([], [], label=dates_str[i])
    lines.append(line)
    visible[line] = False
    line.set_visible(i >= 10)  # Masquer les courbes autres que celles de septembre 2021 au début

# Formatter les dates sur l'axe des abscisses pour afficher uniquement le mois et l'année
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator())

# Ajouter un titre et des étiquettes d'axe
ax.set_title("Évolution de sndMSS, rcvMSS et PMTU")
ax.set_xlabel("Dates")
ax.set_ylabel("Taille")

# Créer une légende pour afficher les mois et années
legend = ax.legend(loc='upper left', title='Mois et année')
legend.set_draggable(True)


# Mettre à jour la visibilité du graphe en fonction de la légende sélectionnée
def update_visibility(event):
    if event.mouseevent.button == 1:  # Bouton gauche de la souris
        line = event.artist
        index = lines.index(line)

        # Masquer toutes les courbes
        for l in lines:
            l.set_visible(False)

        # Afficher les courbes correspondant au mois sélectionné
        if index >= 10:  # Sélection de septembre 2021
            lines[index].set_visible(not visible[line])
            visible[line] = not visible[line]
        else:  # Sélection d'un mois de février ou mars
            start_index = index - (index % 5)  # Récupérer l'index du premier élément du mois
            for i in range(start_index, start_index + 5):
                lines[i].set_visible(not visible[line])
                visible[lines[i]] = not visible[line]

        fig.canvas.draw()


fig.canvas.mpl_connect('pick_event', update_visibility)

plt.show()
