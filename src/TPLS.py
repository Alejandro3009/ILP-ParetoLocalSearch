import random
import copy
from src.solver import calculateFitness, calculateFitnessParallel
from src.model import movements
from src.utils import getTotalDemand, getStateTuple

def createTabuRate(pointsList):
    tabuRate = {}
    for i in range(len(pointsList[0].state)):
        tabuRate[f"{i}opened"] = 0
        tabuRate[f"{i}closed"] = 0
        
    return tabuRate

def createTabuList(cds):
    tabuList = {}
    for cd in cds:
        tabuList[cd.id] = None
    return tabuList

def addTabu(moves, tabuList, addedTabus):
    moveCds = moves.keys()

    for i in moveCds:
        if tabuList[i] is None:
            tabuList[i] = moves[i]
            addedTabus.append(i)

    return tabuList, addedTabus

def isTabu(moves, tabuList):
    moveCds = moves.keys()
    for i in moveCds:
        if moves[i] == tabuList[i]:
            return True
    return False

def removeLastTabu(amountToRemove, tabuList, addedTabus):
    for i in range(amountToRemove):
        tabuList[addedTabus.pop] = None
        if len(tabuList) == 0 or len(addedTabus) == 0:
            break
    return tabuList, addedTabus

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

def AspirationCriteria(tabuState, nonDominatedPoints, foundPoints, cdList, clientList, K, TH, alphaValue):
    exploredTabuPoints = []
    validTabuPoints = []
    tabuStatesToRemove = []

    for state in tabuState:
            for point in foundPoints:
                if point.state == state:
                    tabuStatesToRemove.append(state)
                    exploredTabuPoints.append(point)

    for state in tabuStatesToRemove:
        try:
            tabuState.remove(state)
        except ValueError:
            pass

    tabuPoints, solverTime = calculateFitnessParallel(cdList, clientList, K, TH, tabuState, alphaValue=alphaValue)
    exploredTabuPoints.extend(tabuPoints)
    
    for tabuPoint in exploredTabuPoints:
        validTabuPoint = True
        for point in nonDominatedPoints:
            if tabuPoint.objValueX >= point.objValueX or tabuPoint.objValueY >= point.objValueY:
                validTabuPoint = False
                print("no se cumple criterio de aspiracion")
                break
        if validTabuPoint:
            validTabuPoints.append(tabuPoint)

    return validTabuPoints, solverTime

def getNeighbor(cdList, nonDominatedPoints, tabu, movementSize, totalDemand, K, TH):
    neighborhood = []
    tabuNeighborhood = []
    neighborMovements = []

    for point in nonDominatedPoints:
        i = 0
        openCds = []
        closeCds = []

        for j in range(len(point.state)):
            if point.state[j] == 1:
                openCds.append(j)
            else:
                closeCds.append(j)
        
        while i < 10: # Limitar el número de vecinos a evaluar por cada punto del frente de Pareto
            changedState = list(point.state)

            openAmount = random.randint(0, movementSize)
            closeAmount = random.randint(0, movementSize)

            moves = {}

            if len(closeCds) != 0:
                for j in range(openAmount):
                    cdToMove = random.choice(closeCds)
                    changedState[cdToMove] = 1
                    moves[cdToMove] = 1
                    print(f"abierto: {cdToMove}")

            for j in range(closeAmount):
                print("cerrados")
                cdToMove = random.choice(openCds)
                changedState[cdToMove] = 0
                moves[cdToMove] = 0
                print(f"cerrado: {cdToMove}") 

            if not feasibleSolution(changedState, cdList, totalDemand):
                print("Solución no factible, generando otro vecino...")
                continue

            if isTabu(moves, tabu):
                changedState = tuple(changedState)
                tabuNeighborhood.append(changedState)
                neighborMovements.append(movements(changedState, moves))
            else:
                print("no es tabu")
                changedState = tuple(changedState)
                neighborhood.append(changedState)
                neighborMovements.append(movements(changedState, moves))

            i += 1
            
    return neighborhood, tabuNeighborhood, neighborMovements

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

def checkDominance(pointsList, nonDominatedPoints, neighborMovements):
    pointsToRemove = []
    tabuRate = createTabuRate(pointsList)
    pointsList.extend(nonDominatedPoints)
    foundNonDominated = False #Flag para indicar si se encontró un nuevo punto no dominado

    for evaluatedPoint in pointsList:
        nonDominated = True

        # Se busca el movimiento que se hizo para llegar a ese punto
        movements = None
        for movement in neighborMovements:
            if evaluatedPoint.state == movement.changeState:
                movements = movement.moves
                break
        
        # Se compara con cada punto del frente de Pareto actual
        for referencePoint in pointsList:
            #if referencePoint in pointsToRemove:
            #    print (f"El punto {referencePoint.state} ya está marcado para eliminación, saltando comparación.")
            #    continue

            # Si el nuevo punto domina fuertemente o debilmente a un punto del frente actual, se agrega a la lista de nuevos puntos no dominados 
            # y se elimina el punto dominado del frente actual
            if (evaluatedPoint.objValueX >= referencePoint.objValueX and evaluatedPoint.objValueY >= referencePoint.objValueY) and evaluatedPoint != referencePoint:
                print (f"El punto {evaluatedPoint.state} domina a {referencePoint.state}")
                print (f"Infra: {evaluatedPoint.objValueX} vs {referencePoint.objValueX}, Trans: {evaluatedPoint.objValueY} vs {referencePoint.objValueY}")
                nonDominated = False
                break

        if not nonDominated:
            pointsToRemove.append(evaluatedPoint)
        else:
            foundNonDominated = True
            if movements is not None:
                for move in movements.keys():
                    if movements[move] == 1:
                        tabuRate[f"{move}opened"] += 1
                    else:
                        tabuRate[f"{move}closed"] += 1

    for pointToRemove in pointsToRemove:
        if pointToRemove in pointsList:
            pointsList.remove(pointToRemove)

    return pointsList, foundNonDominated, tabuRate

def removeDuplicateStates(statesList):
    uniquePoints = {}
    
    for state in statesList:
        # Use the state tuple as the unique key
        # This keeps only the first occurrence of each unique state
        if state not in uniquePoints:
            uniquePoints[state] = state
    
    return list(uniquePoints.values())

def removeDuplicatePoints(pointsList):
    uniquePoints = {}
    print("cuantos entraron: " + str(len(pointsList)))
    
    for point in pointsList:
        # Use the state tuple as the unique key
        # This keeps only the first occurrence of each unique state
        if point not in uniquePoints:
            uniquePoints[point] = point
    
    print("cuantos quedaron: " + str(len(uniquePoints)))
    return list(uniquePoints.values())

def tabuLocalParetoSearch(cdList, clientList, K, TH, iterationLimit = 50, movementSize = 3, tabuTenure = 20, amountToAdd = 5, alphaValue = 0.5):
    # 1. Inicialización y obtencion de parametros
    totalDemand = getTotalDemand(clientList)
    nonDominatedPoints = []
    fixCdList = getStateTuple(cdList)
    aux, solverTime = calculateFitness(cdList, clientList, K, TH, [fixCdList], alphaValue)
    nonDominatedPoints.extend(aux)

    foundPoints = []
    foundPoints.extend(nonDominatedPoints)

    i = 0
    iterationwithoutImprovement = 0

    tabu = createTabuList(cdList)
    addedTabus = []

    solverTime = 0

    while i < iterationLimit: 
        # 2. Generar vecinos y remover duplicados
        neighborhood, tabuNeighborhood, neighborMovements = getNeighbor(cdList, nonDominatedPoints, tabu, 
                                                                        movementSize, totalDemand, K, TH)

        neighborhood = removeDuplicateStates(neighborhood)

        tabuNeighborhood = removeDuplicateStates(tabuNeighborhood)

        print (f"Iteración {i+1}/{iterationLimit} - Vecinos generados: {len(neighborhood)} - Vecinos tabu: {len(tabuNeighborhood)}")

        # 3. Evaluar vecinos marcados como tabu con criterio de aspiración
        if len(tabuNeighborhood) > 0:
            validTabuPoints, time = AspirationCriteria(tabuNeighborhood, nonDominatedPoints, foundPoints, cdList, clientList, K, TH, alphaValue)
            solverTime += time

        notFound, alreadyFound = checkIfFound(neighborhood, foundPoints)

        print (f"Iteración {i+1}/{iterationLimit} - Vecinos encontrados: {len(notFound) + len(alreadyFound)}, Vecinos nuevos: {len(notFound)}")

        # 4. Evaluar vecinos no encontrados
        paretoPoints, time  = calculateFitnessParallel(cdList, clientList, K, TH, notFound, alphaValue=alphaValue)
        solverTime += time

        try:
            if len(validTabuPoints) > 0:
                paretoPoints.extend(validTabuPoints)
        except Exception as e:
            pass

        if len(alreadyFound) > 0:
            paretoPoints.extend(alreadyFound)

        for point in paretoPoints:
            print(f"state {point.state}, Infra: {point.objValueX}, Transport: {point.objValueY}")

        # 5. Actualizar el frente de Pareto con la funcion de dominancia
        if len(paretoPoints) > 0:
            nonDominatedPoints, foundNewNonDominated, tabuRate = checkDominance(paretoPoints, nonDominatedPoints, neighborMovements)
        else:
            foundNewNonDominated = False
        
        # 5.1 Criterio de parada por falta de mejora
        if foundNewNonDominated:
            iterationwithoutImprovement = 0
        elif iterationwithoutImprovement >= 3:
            print("No se han encontrado nuevos puntos no dominados en las últimas 10 iteraciones, terminando búsqueda.")
            break
        else:
            iterationwithoutImprovement += 1
            i += 1
            continue
        
        nonDominatedPoints = removeDuplicatePoints(nonDominatedPoints)
        
        # 6. Añadir tabu de los movimientos más frecuentes en los nuevos puntos no dominados encontrados
        sortedTabuRate = sorted(tabuRate.items(), key=lambda x: x[1], reverse=True)
        
        tabuMovesToAdd = {}

        for j in range(amountToAdd):
            move = sortedTabuRate[j]
            moveKey = int(move[0][0])
            moveValue = move[1]
            tabuMovesToAdd[moveKey] = moveValue

        tabu, addedTabus = addTabu(tabuMovesToAdd, tabu, addedTabus)

        if len(addedTabus) > tabuTenure:
            tabu, addedTabus = removeLastTabu(amountToAdd, tabu, addedTabus)
        
        foundPoints.extend(paretoPoints)

        i += 1
    
    return nonDominatedPoints, solverTime