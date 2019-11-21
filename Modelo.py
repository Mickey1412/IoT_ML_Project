import numpy as np
import pandas as pd
from sklearn import svm
from scipy.stats import kurtosis
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


def DataFrameFor(df):
    print(df)
    for data in df:
        print("data: ", data)

# inicializacion de las variables
presion = np.zeros(0)
altitud = np.zeros(0)
humedad = np.zeros(0)
temperatura = np.zeros(0)

count = 1

df = pd.read_csv("Sensado_GYM_Completo.csv", usecols=["Fecha","Presion", "Altitud", "Humedad", "Temperatura","Ocupacion"])
    
dfLow = df[df['Ocupacion'] == 'L']
dfMedium = df[df['Ocupacion'] == 'M']
dfHigh = df[df['Ocupacion'] == 'H']

print(df)

#print(dfLow)
#print(dfMedium)
#print(dfHigh)

DataFrameFor(dfLow)





'''
# For loop para recorrer el csv con intervalos de 10 renglones y generar los calculos
for data in pd.read_csv("Sensado_GYM_Completo.csv", usecols=[
        "Presion", "Altitud", "Humedad", "Temperatura"], chunksize=10):
    # Calculos de la presion
    presion = data["Presion"].values
    presion = presion[np.logical_not(np.isnan(presion))]
    #presionNan = data["Presion"].values
    #print("presion chequeo: ",np.size(presion))

    count = count + 1

    if presion.any():
        #print('**', presion)
        presionMean = presion.mean()
        presionStd = np.std(presion)
        presionKrt = kurtosis(presion)

        # Calculos de la altitud
        altitud = data["Altitud"].values
        #altitudNan = data["Altitud"].values
        altitud = altitud[np.logical_not(np.isnan(altitud))]
        altitudMean = altitud.mean()
        altitudStd = np.std(altitud)
        altitudKrt = kurtosis(altitud)

        # Calculos de la humedad
        humedad = data["Humedad"].values
        #humedadNan = data["Humedad"].values
        humedad = humedad[np.logical_not(np.isnan(humedad))]
        humedadMean = humedad.mean()
        humedadStd = np.std(humedad)
        humedadKrt = kurtosis(humedad)

        # Calculos de la temperatura
        temperatura = data["Temperatura"].values
        #temperaturaNan = data["Temperatura"].values
        temperatura = temperatura[np.logical_not(np.isnan(temperatura))]
        temperaturaMean = temperatura.mean()
        temperaturaStd = np.std(temperatura)
        temperaturaKrt = kurtosis(temperatura)
    else:
        print("something is null")

print("Presion: ",np.size(presion))
print(presion)
print("Altitud: ",np.size(altitud))
print(altitud)
print("Humedad: ",np.size(humedad))
print(humedad)
print("Temperatura: sc",np.size(temperatura))
print(temperatura)
'''