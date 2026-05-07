import random
import copy
from src.solver import calculateFitnessParallel, parallelLinearRelaxation
from src.model import movements
from src.utils import getTotalDemand

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

    if totalCapacity >= totalDemand:
        return True
    else:
        return False

def AspirationCriteria(tabuState, nonDominatedPoints, foundPoints, cdList, clientList, K, TH, alphaValue, lexPoints):
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

    tabuPoints, solverTime = calculateFitnessParallel(cdList, clientList, K, TH, tabuState, alphaValue=alphaValue, lexPoints=lexPoints)

    exploredTabuPoints.extend(tabuPoints)
    
    for tabuPoint in exploredTabuPoints:
        validTabuPoint = True
        for point in nonDominatedPoints:
            if tabuPoint.Infrastructure >= point.Infrastructure or tabuPoint.Transport >= point.Transport:
                validTabuPoint = False
                break
        if validTabuPoint:
            validTabuPoints.append(tabuPoint)

    return validTabuPoints, solverTime

def hybridNeighborGeneration(nonDominatedPoints, cdList, clientList, K, TH, fixingSize, alphaValue, lexPoints, tabu, movementSize):
    totalDemand = getTotalDemand(clientList)
    neighborStates = []
    neighborMovements = []
    tabuNeighborhood = []

    for point in nonDominatedPoints:
        # 1. Rolling the dice: 50% chance for Relaxation, 50% for Standard Tabu
        if random.random() > 1.5:
            # RELAXATION PATH
            relaxNeighbors, relaxMovements, aux = relaxNeighbor(point, cdList, clientList, K, TH, fixingSize, alphaValue, lexPoints, tabu)
            neighborStates.extend(relaxNeighbors)
            neighborMovements.extend(relaxMovements)
            tabuNeighborhood.extend(aux)
        else:
            # STANDARD TABU PATH
            standardNeighbors, tabuNeighbors, standardMovements = getNeighbor(cdList, [point], tabu, movementSize, totalDemand, K, TH)
            neighborStates.extend(standardNeighbors)
            neighborMovements.extend(standardMovements)
            tabuNeighborhood.extend(tabuNeighbors)
    
    return neighborStates, neighborMovements, tabuNeighborhood

# TPLS_MPS.py
def relaxNeighbor(point, cdList, clientList, K, TH, fixingSize, alphaValue, lexPoints, tabu):
    neighborStates = []
    neighborMovements = []
    i = 0
    
    while i < 10:
        # RELAXATION PATH
        relaxResult, _ = parallelLinearRelaxation([point.state], cdList, clientList, K, TH, 
                                                    fixingSize, alphaValue=alphaValue, 
                                                    lexPoints=lexPoints, tabuList=tabu)
        if relaxResult:
            actualState = tuple(relaxResult[0])
            # We record the movement even for relaxation to update Tabu frequency later
            moves = {i: actualState[i] for i in range(len(actualState)) if actualState[i] != point.state[i]}
            neighborStates.append(actualState)
            neighborMovements.append(movements(actualState, moves))
        
        i += 1

    return neighborStates, neighborMovements, []

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

            for j in range(closeAmount):
                cdToMove = random.choice(openCds)
                changedState[cdToMove] = 0
                moves[cdToMove] = 0

            #if not feasibleSolution(changedState, cdList, totalDemand):
            #    continue

            if isTabu(moves, tabu):
                changedState = tuple(changedState)
                tabuNeighborhood.append(changedState)
                neighborMovements.append(movements(changedState, moves))
            else:
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

    for evaluatedPoint in pointsList:
        nonDominated = True

        # Se busca el movimiento que se hizo para llegar a ese punto
        movements = None
        for movement in neighborMovements:
            if evaluatedPoint.state == movement.changeState:
                movements = movement.moves
                break
        
        # Se compara con cada punto del frente de Pareto actual
        # Si el punto de referencia fue dominado anteriormente, no se usa durante las comparaciones
        for referencePoint in pointsList:
            if referencePoint in pointsToRemove:
                continue

            # Si el nuevo punto domina fuertemente o debilmente a un punto del frente actual, se agrega a la lista de nuevos puntos no dominados 
            # y se elimina el punto dominado del frente actual
            if (evaluatedPoint.Infrastructure >= referencePoint.Infrastructure and evaluatedPoint.Transport >= referencePoint.Transport) and evaluatedPoint != referencePoint:
                nonDominated = False
                break

        if not nonDominated:
            pointsToRemove.append(evaluatedPoint)
        else:
            if movements is not None:
                for move in movements.keys():
                    if movements[move] == 1:
                        tabuRate[f"{move}opened"] += 1
                    else:
                        tabuRate[f"{move}closed"] += 1

    for pointToRemove in pointsToRemove:
        if pointToRemove in pointsList:
            pointsList.remove(pointToRemove)

    return pointsList, tabuRate

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
    
    for point in pointsList:
        # Use the state tuple as the unique key
        # This keeps only the first occurrence of each unique state
        if point not in uniquePoints:
            uniquePoints[point] = point

    return list(uniquePoints.values())

def MultiPointParetoSearch(initialState, cdList, clientList, K, TH, lexPoints, iterationLimit = 50, movementSize = 3, tabuTenure = 20, amountToAdd = 5, alphaValue = 0.5):
    # 1. Inicialización y obtencion de parametros
    totalDemand = getTotalDemand(clientList)
    nonDominatedPoints = []
    aux, solverTime = calculateFitnessParallel(cdList, clientList, K, TH, initialState, alphaValue=alphaValue, lexPoints=lexPoints)
    nonDominatedPoints.extend(aux)

    foundPoints = []
    foundPoints.extend(nonDominatedPoints)

    i = 0
    iterationwithoutImprovement = 0

    tabu = createTabuList(cdList)
    addedTabus = []

    solverTime = 0
    
    stopped = [False, None]

    while i < iterationLimit: 
        # 2. Generar vecinos y remover duplicados
        
        print (f"Iteración {i+1}/{iterationLimit} - Generando vecinos...")

        neighborhood, neighborMovements, tabuNeighborhood = hybridNeighborGeneration(nonDominatedPoints, cdList, clientList, K, TH, fixingSize=movementSize, 
                                                                                     alphaValue=alphaValue, lexPoints=lexPoints, tabu=tabu, movementSize=movementSize)

        neighborhood = removeDuplicateStates(neighborhood)

        if len(tabuNeighborhood) == 0:
            tabuNeighborhood = removeDuplicateStates(tabuNeighborhood)

        print (f"Iteración {i+1}/{iterationLimit} - Vecinos generados: {len(neighborhood)} - Vecinos tabu: {len(tabuNeighborhood)}")

        # 3. Evaluar vecinos marcados como tabu con criterio de aspiración
        if len(tabuNeighborhood) > 0:
            validTabuPoints, time = AspirationCriteria(tabuNeighborhood, nonDominatedPoints, foundPoints, cdList, clientList, K, TH, alphaValue, lexPoints)
            solverTime += time
        else :
            validTabuPoints = []

        notFound, alreadyFound = checkIfFound(neighborhood, foundPoints)

        print (f"Iteración {i+1}/{iterationLimit} - Vecinos encontrados: {len(notFound) + len(alreadyFound)}, Vecinos nuevos: {len(notFound)}")

        # 4. Evaluar vecinos no encontrados
        paretoPoints, time  = calculateFitnessParallel(cdList, clientList, K, TH, notFound, alphaValue=alphaValue, lexPoints=lexPoints)
            
        solverTime += time

        # Capture the states of the front BEFORE adding already known points
        previousFrontStates = set(p.state for p in nonDominatedPoints)

        # Only consider points that were NOT previously found in this specific search
        newlyEvaluatedPoints = []
        newlyEvaluatedPoints.extend(paretoPoints) # These come from 'notFound'
        if len(validTabuPoints) > 0:
            newlyEvaluatedPoints.extend(validTabuPoints)

        # Now update the front using EVERYTHING (new + old) to maintain correctness
        allPotentialPoints = newlyEvaluatedPoints + alreadyFound

        allPotentialPoints = removeDuplicatePoints(allPotentialPoints)

        nonDominatedPoints, tabuRate = checkDominance(allPotentialPoints, nonDominatedPoints, neighborMovements)

        nonDominatedPoints = removeDuplicatePoints(nonDominatedPoints)

        # --- THE FIX ---
        # Check if any of the NEWLY evaluated points made it into the front
        currentFrontStates = set(p.state for p in nonDominatedPoints)
        # An improvement only counts if a NEW state was added that isn't in the "history"
        actualImprovement = any(p.state in currentFrontStates for p in newlyEvaluatedPoints)
        frontChanged = not currentFrontStates.issubset(previousFrontStates)

        if actualImprovement and frontChanged:
            iterationwithoutImprovement = 0
            print(f"Iteración {i+1}/{iterationLimit} - Nuevo punto no dominado encontrado! Total en el frente: {len(nonDominatedPoints)}")
        elif iterationwithoutImprovement >= 5:
            print("No se han encontrado nuevos puntos no dominados en las últimas 5 iteraciones, terminando búsqueda.")
            stopped = [True, i]
            nonDominatedPoints = removeDuplicatePoints(nonDominatedPoints)
            break
        else:
            iterationwithoutImprovement += 1
            i += 1
            print(f"Iteración {i}/{iterationLimit} - No se encontraron nuevos puntos no dominados. Iteraciones sin mejora: {iterationwithoutImprovement}")
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
    
    return nonDominatedPoints, solverTime, stopped