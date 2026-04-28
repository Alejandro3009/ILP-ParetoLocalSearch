import numpy as np
import random
from src.utils import getTotalDemand, getStateTuple

# Movements selection

def generateInitialSolution(selectedStrategy, cds, clients):
    match selectedStrategy:
        case 1:
            return dualPriorityList(cds, clients, amount=5)
        case 0:
            return randomSolution(cds, clients)
        case _:
            raise ValueError("Invalid selection strategy")
        
# random solution

def randomSolution(cdList, clientList):
    for client in clientList:
        chosen = random.choice(cdList)
        client.assignedCd = chosen.id
        chosen.open = True
        chosen.assignedDemand += client.demand
        chosen.assignedVariance += client.variance

    aux = [getStateTuple(cdList)]
    return aux

# Dual-Priority List

def getPriorityTransportList(cds, clients):
    # ordenamos los centros por eficiencia en transporte a clientes
    transportEfficiency = []
    for cd in cds:
        totalTransportCost = sum([client.transportCost[cd.id] for client in clients])
        avgTransportCost = totalTransportCost / len(clients)
        transportEfficiency.append((cd, avgTransportCost / cd.capacity)) # eficiencia = coste total de transporte / capacidad del centro

    return sorted(transportEfficiency, key=lambda x: x[1], reverse=True)

def getPriorityCostList(cds):
    # ordenamos los centros por eficiencia en coste de apertura
    openingEfficiency = []
    for cd in cds:
         openingEfficiency.append((cd, cd.fixedCost / cd.capacity)) # eficiencia = coste fijo / capacidad

    return sorted(openingEfficiency, key=lambda x: x[1], reverse=True)

def dualPriorityList(cds, clients, amount):
    # arreglo de centros priorizados por eficiencia en transporte a clientes
    priorityTransportList = getPriorityTransportList(cds, clients)
    # arreglo de centros priorizados por eficiencia en coste de apertura
    priorityCostList = getPriorityCostList(cds)

    # inizializacion
    initialStates = []
    if amount == 1:
        ponderates = [0.5]
    else:
        ponderates = np.linspace(0, 1, amount)    

    for alpha in ponderates:
        # calculamos la puntuación combinada para cada centro
        cdList = cds.copy()
        combinedEfficiency = []
        for cd in cdList:
            transportScore = next((x[1] for x in priorityTransportList if x[0].id == cd.id), 0)
            costScore = next((x[1] for x in priorityCostList if x[0].id == cd.id), 0)
            combinedScore = alpha * transportScore + (1 - alpha) * costScore
            combinedEfficiency.append((cd, combinedScore))

        # ordenamos los centros por puntuación combinada
        combinedEfficiency.sort(key=lambda x: x[1], reverse=True)

        # seleccionamos el centro con la mejor puntuación combinada
        totalDemand = getTotalDemand(clients)
        totalCapacity = 0

        for cd, _ in combinedEfficiency:
            if totalCapacity >= totalDemand * 1.5:
                break

            totalCapacity += cd.capacity
            cd.open = True

        initialStates.append(getStateTuple(cdList))

    return initialStates