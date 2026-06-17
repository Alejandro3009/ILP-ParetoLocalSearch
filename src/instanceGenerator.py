import numpy as np
import random
import math
import json
from src.model import cd
import src.model as model

def generateInstance(numClients, planeSize=100, unitTransportCost=2.12, seed=None, clustered=False, numClusters=3, stdDev=10, baseMinCost=15000, maxPremium=25000, 
                   baseMinCapacity=1100, maxCapacity=2500):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    cdList = []
    clientList = []

    if clustered:
        centroids = [(random.uniform(0, planeSize), random.uniform(0, planeSize)) for _ in range(numClusters)]

    # Calcular la cantidad de CDs (la mitad de los clientes)
    numCds = max(1, numClients // 2)

    # 1. Generar Clientes
    for j in range(numClients):
        # Ajustado a los rangos de la instancia proporcionada
        demand = np.random.randint(30, 151)  # 101 para que incluya el 100
        variance = np.random.randint(25, 51)  # 23 para que incluya el 22

        if clustered:
            centerX, centerY = random.choice(centroids)
                
            # Spawn client around center with a standard deviation (tightness of the city)
            posX = np.clip(np.random.normal(centerX, stdDev), 0, planeSize)
            posY = np.clip(np.random.normal(centerY, stdDev), 0, planeSize)   
        else:
            posX = np.random.uniform(0, planeSize)
            posY = np.random.uniform(0, planeSize)

        # Los índices en tu data original empiezan en 0, así que usamos 'j' directo
        clientId = j

        newClient = model.client(clientId, demand, variance, posX, posY)

        clientList.append(newClient)

    # 2. Generar Centros de Distribución (CDs)
    for i in range(numCds):
        # Ajustado a los rangos de la instancia proporcionada
             
        reorderCost = np.random.randint(180, 221)  # 221 para que incluya el 220
        holdingCost = round(np.random.uniform(2.0, 2.4), 2) # Redondeado a 2 decimales
        leadTime = np.random.randint(6, 16)        # 16 para que incluya el 15
        replenishmentCost = np.random.randint(500, 600)

        posX = np.random.uniform(0, planeSize)
        posY = np.random.uniform(0, planeSize)

        if clustered:            
            # Find distance to the closest city center
            minDistToCenter = min([math.sqrt((posX - cx)**2 + (posY - cy)**2) for cx, cy in centroids])
            
            # Higher cost if distance is small (Center of the cluster)
            # We normalize distance: 0 distance = max cost, planeSize distance = min cost
            proximityFactor = max(0, 1 - (minDistToCenter / (planeSize / 2)))
            fixedCost = int(baseMinCost + (proximityFactor * maxPremium))
            # Capacity is also influenced by proximity to the cluster center, with more capacity for those farther from the center
            capacityRange = maxCapacity - baseMinCapacity
            capacity = int(baseMinCapacity + ((1 - proximityFactor) * capacityRange))  # Up to 50% more capacity for central CDs
        else:
            fixedCost = np.random.randint(15000, 42000)
            capacity = np.random.randint(baseMinCapacity, maxCapacity)
        # Los índices en tu data original empiezan en 0, así que usamos 'i' directo
        cdId = i
        cdList.append(cd(cdId, capacity, fixedCost, reorderCost, holdingCost, leadTime, replenishmentCost, posX, posY))

    # 3. Calcular costos de transporte
    for client in clientList:
        for warehouse in cdList:
            distance = math.sqrt((warehouse.posX - client.posX)**2 + (warehouse.posY - client.posY)**2)

            # Multiplicamos por el costo unitario y redondeamos al entero más cercano
            calculatedCost = int(round(distance * unitTransportCost))

            # Guardamos el costo en el diccionario usando el ID del CD como llave
            client.transportCost[warehouse.id] = calculatedCost

    return cdList, clientList, centroids if clustered else None

# Función para guardar el string en un archivo físico
def saveDatFile(fileContent, fileName="instancia.dat"):
    # Abrimos el archivo en modo escritura ('w' = write)
    with open(fileName, 'w', encoding='utf-8') as file:
        file.write(fileContent)

    print(f"Archivo AMPL guardado exitosamente como: {fileName}")

def saveVisualData(cdList, clientList, centroids, fileName="instance_coords.json"):
    # Creamos un diccionario base para estructurar nuestros datos
    visualData = {
        "cds": [],
        "clients": [],
        "centroids": []
    }

    # Extraer información de los CDs
    for cd in cdList:
        visualData["cds"].append({
            "id": cd.id,
            "posX": round(cd.posX, 2),
            "posY": round(cd.posY, 2)
        })

    # Extraer información de los Clientes
    for client in clientList:
        visualData["clients"].append({
            "id": client.id,
            "posX": round(client.posX, 2),
            "posY": round(client.posY, 2)
        })
    
    # Extraer información de los Centroides (si existen)
    if centroids is not None:
        for idx, (cx, cy) in enumerate(centroids):
            visualData["centroids"].append({
                "id": idx,
                "posX": round(cx, 2),
                "posY": round(cy, 2)
            })

    # Escribir el diccionario en un archivo JSON
    with open(fileName, 'w', encoding='utf-8') as jsonFile:
        json.dump(visualData, jsonFile, indent=4)

    print(f"Coordenadas guardadas exitosamente en {fileName}")