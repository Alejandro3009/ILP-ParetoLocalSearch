import random
from src.model import cd, client, paretoPoint
from src.utils import getStateTuple, getTotalDemand
from src.solver import calculateFitness

def getNeighborhood(cdList, paretoPoints, totalDemand, movementSize):
    neighborhood = []
    for point in paretoPoints:
        i=0
        openCds = []
        closeCds = []

        for j in range(len(point.state)):
            if point.state[j] == 1:
                openCds.append(j)
            else:
                closeCds.append(j)
        
        while True:
            changedState = list(point.state)

            openAmount = random.randint(0, movementSize)
            closeAmount = random.randint(0, movementSize)

            for j in range(openAmount):
                cdToMove = random.choice(closeCds)
                changedState[cdToMove] = 1
                print(f"abierto: {cdToMove}")

            for j in range(closeAmount):
                print("cerrados")
                cdToMove = random.choice(openCds)
                changedState[cdToMove] = 0
                print(f"cerrado: {cdToMove}") 

            if feasibleSolution(changedState, cdList, totalDemand):
                changedState = tuple(changedState)
                neighborhood.append(changedState)
                i += 1
            else:
                i += 1

            if i >= 5: # Limitar el número de intentos para generar vecinos
                break

    return neighborhood

def feasibleSolution(state, cdList, totalDemand):
    totalCapacity = 0
    for i in range(len(state)):
        if state[i] == 1:
            totalCapacity += cdList[i].capacity
    
    print (f"Total Capacity: {totalCapacity}, Total Demand + Variance: {totalDemand}")

    if totalCapacity >= totalDemand:
        print("Solución factible")
        return True
    else:
        print("Solución no factible")
        return False

def checkIfFound(neighborState, exploredPoints):
    alreadyFound = []
    toRemove = []
    aux = []
    for i in range(len(neighborState)):
        for point in exploredPoints:
            if point.state == neighborState[i]:
                alreadyFound.append(point)
                toRemove.append(i)
                break
    
    for index in toRemove:
        aux.append(neighborState[index])

    for item in aux:
        neighborState.remove(item)

    return neighborState, alreadyFound
    
def checkDominance(pointsList, nonDominatedPoints):
    pointsToRemove = []
    pointsList.extend(nonDominatedPoints)
    foundNonDominated = False #Flag para indicar si se encontró un nuevo punto no dominado

    for evaluatedPoint in pointsList:
        nonDominated = True

        # Se compara con cada punto del frente de Pareto actual
        for referencePoint in pointsList:
            # Si el nuevo punto domina fuertemente o debilmente a un punto del frente actual, se agrega a la lista de nuevos puntos no dominados 
            # y se elimina el punto dominado del frente actual
            if (evaluatedPoint.objValueX >= referencePoint.objValueX and evaluatedPoint.objValueY >= referencePoint.objValueY) and evaluatedPoint != referencePoint:
                nonDominated = False
                break

        if not nonDominated:
            foundNonDominated = True
            pointsToRemove.append(evaluatedPoint)

    for pointToRemove in pointsToRemove:
        if pointToRemove in pointsList:
            pointsList.remove(pointToRemove)

    return pointsList, foundNonDominated

def removeDuplicatePoints(pointsList):
    uniquePoints = {}
    
    for point in pointsList:
        # Use the state tuple as the unique key
        # This keeps only the first occurrence of each unique state
        if point.state not in uniquePoints:
            uniquePoints[point.state] = point
    
    return list(uniquePoints.values())

def removeDuplicateStates(statesList):
    uniquePoints = {}
    
    for state in statesList:
        # Use the state tuple as the unique key
        # This keeps only the first occurrence of each unique state
        if state not in uniquePoints:
            uniquePoints[state] = state
    
    return list(uniquePoints.values())
                
def paretoLocalSearch(cdList, clientList, K, TH, iterationLimit = 50, movementSize = 3):
    totalDemand = getTotalDemand(clientList)
    nonDominatedPoints = []
    fixCdList = getStateTuple(cdList)
    nonDominatedPoints.extend(calculateFitness(cdList, clientList, K, TH, [fixCdList]))

    foundStates = []
    foundStates.extend(nonDominatedPoints)

    i = 0
    iterationwithoutImprovement = 0

    while i < iterationLimit:
        neighborhood = getNeighborhood(cdList, nonDominatedPoints, totalDemand, movementSize)

        neighborhood = removeDuplicateStates(neighborhood)

        print (f"Iteración {i+1}/{iterationLimit} - Vecinos generados: {len(neighborhood)}")
        
        notFound, alreadyFound = checkIfFound(neighborhood, foundStates)

        print (f"Iteración {i+1}/{iterationLimit} - Vecinos encontrados: {len(notFound) + len(alreadyFound)}, Vecinos nuevos: {len(notFound)}")

        paretoPoints = calculateFitness(cdList, clientList, K, TH, notFound)

        aux = paretoPoints.copy()

        paretoPoints.extend(alreadyFound)

        nonDominatedPoints, flag = checkDominance(paretoPoints, nonDominatedPoints)

        nonDominatedPoints = removeDuplicatePoints(nonDominatedPoints)

        if flag:
            iterationwithoutImprovement = 0
        else:
            iterationwithoutImprovement += 1

        if iterationwithoutImprovement >= 3: # Si no se encuentra un nuevo punto no dominado en 3 iteraciones consecutivas, se detiene la búsqueda
            break
        
        foundStates.extend(aux)
        
        print(f"Iteración {i+1}/{iterationLimit} - Longitud de lista de puntos encontrados: {len(foundStates)}")

        i += 1
    
    return nonDominatedPoints