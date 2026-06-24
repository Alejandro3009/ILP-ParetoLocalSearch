import numpy as np
import random
from src.utils import getTotalDemand
from src.model import modelo
from amplpy import AMPL

# Movements selection

def generateInitialSolution(selectedStrategy, currentInstance):
    match selectedStrategy:
        case "dualPriorityList":
            return dualPriorityList(currentInstance, amount=5)
        case "random":
            return randomSolution(currentInstance, amount=5)
        case _:
            raise ValueError("Estrategia de generación de solución inicial no reconocida.")
        
# random solution

def randomSolution(instanceContent, amount = 5):
    tempAmpl = AMPL()
    tempAmpl.eval("reset;")
    tempAmpl.eval(modelo)
    tempAmpl.eval(instanceContent)

    cdsCapacity = tempAmpl.getParameter("Cap").getValues().toDict()
    clientsDemand = tempAmpl.getParameter("d").getValues().toDict()
    clientsVariance = tempAmpl.getParameter("u").getValues().toDict()

    cds = tempAmpl.getSet("I").getValues().toList()

    tempAmpl.close() # Cerramos la instancia temporal de AMPL para liberar recursos
    
    finalSolutions = []

    for i in range(amount):
        demandAssignation = {}
        j = 0
        for cd in cds:
            demandAssignation[cd] = 0
        
        while j < len(clientsDemand):
            clientDemand = clientsDemand[j]
            clientVariance = clientsVariance[j]
            assigned = False

            while not assigned:
                cdIndex = random.randint(0, len(cdsCapacity) - 1)
                if demandAssignation[cdIndex] + clientDemand + clientVariance <= cdsCapacity[cdIndex]:
                    demandAssignation[cdIndex] += clientDemand
                    demandAssignation[cdIndex] += clientVariance
                    assigned = True
            
            j += 1
        
        solution = [0 for i in cds]

        for cd in demandAssignation:
            if demandAssignation[cd] > 0:
                solution[cd] = 1
        
        finalSolutions.append(tuple(solution))

    return finalSolutions

# Dual-Priority List

def getPriorityTransportList(instanceContent):
    tempAmpl = AMPL()
    tempAmpl.eval("reset;")
    tempAmpl.eval(modelo)
    tempAmpl.eval(instanceContent)

    clientsTransportCost= tempAmpl.getParameter("TC").getValues().toDict()
    cdsCapacity = tempAmpl.getParameter("Cap").getValues().toDict()

    clients = tempAmpl.getSet("J").getValues().toList()
    cds = tempAmpl.getSet("I").getValues().toList()

    tempAmpl.close() # Cerramos la instancia temporal de AMPL para liberar recursos

    # ordenamos los centros por eficiencia en transporte a clientes
    transportEfficiency = []
    for cd in cds:
        totalTransportCost = sum([clientsTransportCost[cd, client] for client in clients])
        avgTransportCost = totalTransportCost / len(clients)
        transportEfficiency.append((cd, avgTransportCost / cdsCapacity[cd])) # eficiencia = coste total de transporte / capacidad del centro

    return sorted(transportEfficiency, key=lambda x: x[1], reverse=True)

def getPriorityCostList(instanceContent):
    tempAmpl = AMPL()
    tempAmpl.eval("reset;")
    tempAmpl.eval(modelo)
    tempAmpl.eval(instanceContent)

    cdsFixedCost = tempAmpl.getParameter("F").getValues()
    cdsCapacity = tempAmpl.getParameter("Cap").getValues()

    cds = tempAmpl.getSet("I").getValues().toList()

    tempAmpl.close() # Cerramos la instancia temporal de AMPL para liberar recursos

    # ordenamos los centros por eficiencia en coste de apertura
    openingEfficiency = []
    for cd in cds:
         openingEfficiency.append((cd, cdsFixedCost[cd] / cdsCapacity[cd])) # eficiencia = coste fijo / capacidad

    return sorted(openingEfficiency, key=lambda x: x[1], reverse=True)

def dualPriorityList(instanceContent, amount):
    # arreglo de centros priorizados por eficiencia en transporte a clientes
    priorityTransportList = getPriorityTransportList(instanceContent)
    # arreglo de centros priorizados por eficiencia en coste de apertura
    priorityCostList = getPriorityCostList(instanceContent)

    # inizializacion
    initialStates = []
    if amount == 1:
        ponderates = [0.5]
    else:
        ponderates = np.linspace(0, 1, amount)    

    tempAmpl = AMPL()
    tempAmpl.eval("reset;")
    tempAmpl.eval(modelo)
    tempAmpl.eval(instanceContent)

    cds = tempAmpl.getSet("I").getValues().toList()
    cdsCapacity = tempAmpl.getParameter("Cap").getValues().toDict()

    totalDemand = getTotalDemand(instanceContent)

    tempAmpl.close() # Cerramos la instancia temporal de AMPL para liberar recursos

    for alpha in ponderates:
        # calculamos la puntuación combinada para cada centro
        cdList = cds.copy()
        combinedEfficiency = []
        for cd in cdList:
            transportScore = next((x[1] for x in priorityTransportList if x[0] == cd), 0)
            costScore = next((x[1] for x in priorityCostList if x[0] == cd), 0)
            combinedScore = alpha * transportScore + (1 - alpha) * costScore
            combinedEfficiency.append((cd, combinedScore))

        # ordenamos los centros por puntuación combinada
        combinedEfficiency.sort(key=lambda x: x[1], reverse=True)

        # seleccionamos el centro con la mejor puntuación combinada     
        totalCapacity = 0
        solution = [0 for i in cds]

        for cd, _ in combinedEfficiency:
            if totalCapacity >= totalDemand:
                break

            totalCapacity += cdsCapacity[cd]
            solution[cd] = 1

        initialStates.append(tuple(solution))

    return initialStates