"""
Startercode bij Lesbrief: Machine Learning, CMTPRG01-9

Deze code is geschreven in Python3

Benodigde libraries:
- NumPy
- SciPy
- matplotlib
- sklearn

"""
from machinelearningdata import Machine_Learning_Data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def extract_from_json_as_np_array(key, json_data):
    """ helper functie om data uit de json te halen en om te zetten naar numpy array voor sklearn"""
    data_as_array = []
    for p in json_data:
        data_as_array.append(p[key])

    return np.array(data_as_array)


STUDENTNUMMER = "0906233" # TODO: aanpassen aan je eigen studentnummer

assert STUDENTNUMMER != "1234567", "Verander 1234567 in je eigen studentnummer"

print("STARTER CODE")

# maak een data-object aan om jouw data van de server op te halen
data = Machine_Learning_Data(STUDENTNUMMER)


# UNSUPERVISED LEARNING
def unsupervisedLearning():
    """kmeans """
    # haal clustering data op
    kmeans_training = data.clustering_training()

    # extract de x waarden
    X = extract_from_json_as_np_array("x", kmeans_training)

    #print(X)

    # slice kolommen voor plotten (let op, dit is de y voor de y-as, niet te verwarren met een y van de data)
    x = X[...,0]
    y = X[...,1]

    cluster_amount = 3
    cluster_Colors = ["r.", "g.", "b.", "y.", "c.", "m.", "k."] 

    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    kmeans = KMeans(n_clusters = cluster_amount, init = 'k-means++', n_init = 10, max_iter = 300)

    # Computes k-means clustering
    kmeans.fit(X)

    # Coordinates of cluster centers
    clusters = kmeans.cluster_centers_

    # Number of iterations that were run
    iterations = kmeans.n_iter_

    # Each point has a lable relevant to its cluster, use this to draw clusters with the same colors
    labels = kmeans.labels_

    print(clusters, iterations) 

    # teken de punten
    for i in range(len(x)):
        plt.plot(X[i][0], X[i][1], cluster_Colors[labels[i]], markersize=8)
        # plt.plot(x[i], y[i], 'k.') # k = zwart

    for i in range(len(clusters)):
        """ Draw the cluster centroids"""
        plt.scatter(clusters[:, 0], clusters[:, 1], marker="x", c="k", s=50, zorder=10) 

    plt.axis([min(x), max(x), min(y), max(y)])
    plt.show()

# SUPERVISED LEARNING

# haal data op voor classificatie
classification_training = data.classification_training()

# extract de data x = array met waarden, y = classificatie 0 of 1
X = extract_from_json_as_np_array("x", classification_training)

# dit zijn de werkelijke waarden, daarom kan je die gebruiken om te trainen
Y = extract_from_json_as_np_array("y", classification_training)


# TODO: leer de classificaties

# TODO: voorspel na het trainen de Y-waarden (je gebruikt hiervoor dus niet meer de
#       echte Y-waarden, maar enkel de X en je getrainde classifier) en noem deze
#       bijvoordeeld Y_predict

# TODO: vergelijk Y_predict met de echte Y om te zien hoe goed je getraind hebt


# haal data op om te testen
classification_test = data.classification_test()
# testen doen we 'geblinddoekt' je krijgt nu de Y's niet
X_test = extract_from_json_as_np_array("x", classification_test)

# TODO: voorspel na nog een keer de Y-waarden, en plaats die in de variabele Z
#       je kunt nu zelf niet testen hoe goed je het gedaan hebt omdat je nu
#       geen echte Y-waarden gekregen hebt.
#       onderstaande code stuurt je voorspelling naar de server, die je dan
#       vertelt hoeveel er goed waren.

Z = np.zeros(100) # dit is een gok dat alles 0 is... kan je zelf voorspellen hoeveel procent er goed is?

# stuur je voorspelling naar de server om te kijken hoe goed je het gedaan hebt
classification_test = data.classification_test(Z.tolist()) # tolist zorgt ervoor dat het numpy object uit de predict omgezet wordt naar een 'normale' lijst van 1'en en 0'en
print("Classificatie accuratie (test): " + str(classification_test))

