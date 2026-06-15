import random
import copy
from amplpy import AMPL
from src.solver import calculateFitnessParallel, parallelLinearRelaxation
from src.model import movements, modelo
from src.utils import getTotalDemand

def createTabuList(instanceContent):
    tempAmpl = AMPL()
    tempAmpl.eval("reset;")
    tempAmpl.eval(modelo)
    tempAmpl.eval(instanceContent)

    cds = tempAmpl.getSet("I").getValues().toList()

    tempAmpl.close() # Cerramos la instancia temporal de AMPL para liberar recursos

    tabuList = {}
    for cd in cds:
        tabuList[cd] = None
    return tabuList

def createTabuRate(cds):
    tabuRate = {}
    
    # Defensive check: if cds is empty, handle gracefully to prevent IndexError
    if len(cds) == 0:
        return tabuRate
        
    for i in range(len(cds[0].state)):
        tabuRate[f"{i}opened"] = 0
        tabuRate[f"{i}closed"] = 0
    
    return tabuRate

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

def AspirationCriteria(instance, tabuState, nonDominatedPoints, foundPoints, maxWorkers, alphaValue, lexPoints):
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

    tabuPoints, solverTime, solverIterations, solverNodes = calculateFitnessParallel(instance, tabuState, max_workers=maxWorkers, alphaValue=alphaValue, lexPoints=lexPoints)

    exploredTabuPoints.extend(tabuPoints)
    
    for tabuPoint in exploredTabuPoints:
        validTabuPoint = True
        for point in nonDominatedPoints:
            if tabuPoint.Infrastructure >= point.Infrastructure or tabuPoint.Transport >= point.Transport:
                validTabuPoint = False
                break
        if validTabuPoint:
            validTabuPoints.append(tabuPoint)

    return validTabuPoints, solverTime, solverIterations, solverNodes

def selectOperator(movementsOperators):
    if isinstance(movementsOperators, str):
        return movementsOperators

    methods = [item["method"] for item in movementsOperators]
    probabilities = [item["prob"] for item in movementsOperators]
    return random.choices(methods, weights=probabilities, k=1)[0]

def hybridNeighborGeneration(nonDominatedPoints, instance, movementsOperators, neiborsGenerated, fixingSize, maxWorkers, alphaValue, lexPoints, tabu, movementSize):
    neighborStates = []
    neighborMovements = []
    tabuNeighborhood = []
    solverData = [0,0]

    operator = selectOperator(movementsOperators)
    
    # 1. Rolling the dice: 50% chance for Relaxation, 50% for Standard Tabu
    if operator == "relaxed":
        # RELAXATION PATH
        relaxNeighbors, relaxMovements, aux, callIterations, callNodes = relaxNeighbor(instance, nonDominatedPoints, neiborsGenerated, fixingSize, maxWorkers, alphaValue, lexPoints, tabu)
        neighborStates.extend(relaxNeighbors)
        neighborMovements.extend(relaxMovements)
        tabuNeighborhood.extend(aux)
        solverData[0] += callIterations
        solverData[1] += callNodes
    elif operator == "random":
        # STANDARD TABU PATH
        standardNeighbors, tabuNeighbors, standardMovements = getNeighbor(nonDominatedPoints, tabu, movementSize, neiborsGenerated)
        neighborStates.extend(standardNeighbors)
        neighborMovements.extend(standardMovements)
        tabuNeighborhood.extend(tabuNeighbors)
    
    return neighborStates, neighborMovements, tabuNeighborhood, solverData

def relaxNeighbor(instance, point, neiborsGenerated, fixingSize, maxWorkers, alphaValue, lexPoints, tabu):
    neighborStates = []
    neighborMovements = []
    i = 0
    
    while i < neiborsGenerated:
        # RELAXATION PATH
        relaxResult, _, iterations, nodes= parallelLinearRelaxation(instance, [point.state], fixingSize, max_workers=maxWorkers, alphaValue=alphaValue, 
                                                    lexPoints=lexPoints, tabuList=tabu)
        if relaxResult:
            actualState = tuple(relaxResult[0])
            # We record the movement even for relaxation to update Tabu frequency later
            moves = {i: actualState[i] for i in range(len(actualState)) if actualState[i] != point.state[i]}
            neighborStates.append(actualState)
            neighborMovements.append(movements(actualState, moves))
        
        i += 1

    return neighborStates, neighborMovements, [], iterations, nodes

def getNeighbor(point, tabu, movementSize, neiborsGenerated):
    neighborhood = []
    tabuNeighborhood = []
    neighborMovements = []
    i = 0
    openCds = []
    closeCds = []

    for j in range(len(point.state)):
        if point.state[j] == 1:
            openCds.append(j)
        else:
            closeCds.append(j)
        
    while i < neiborsGenerated: # Limitar el número de vecinos a evaluar por cada punto del frente de Pareto
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

            # Si el nuevo punto domina fuertemente o debilmente a un punto del frente actual, se agrega a la lista de nuevos puntos no dominados 
            # y se elimina el punto dominado del frente actual
            is_worse_or_equal = (evaluatedPoint.Infrastructure >= referencePoint.Infrastructure and 
                                 evaluatedPoint.Transport >= referencePoint.Transport)
            is_strictly_worse = (evaluatedPoint.Infrastructure > referencePoint.Infrastructure or 
                                 evaluatedPoint.Transport > referencePoint.Transport)
            
            if is_worse_or_equal and is_strictly_worse and evaluatedPoint.state != referencePoint.state:
                nonDominated = False
                break

            if (evaluatedPoint.Infrastructure == referencePoint.Infrastructure and 
                evaluatedPoint.Transport == referencePoint.Transport) and evaluatedPoint.state != referencePoint.state:
                if evaluatedPoint.state > referencePoint.state:
                    nonDominated = False
                    break

            #if (evaluatedPoint.Infrastructure >= referencePoint.Infrastructure and evaluatedPoint.Transport >= referencePoint.Transport) and evaluatedPoint.state != referencePoint.state:
                #print (f"El punto {evaluatedPoint.state} domina a {referencePoint.state}")
                #print (f"Infra: {evaluatedPoint.Infrastructure} vs {referencePoint.Infrastructure}, Trans: {evaluatedPoint.Transport} vs {referencePoint.Transport}")
            #    nonDominated = False
            #    break

        if not nonDominated:
            pointsToRemove.append(evaluatedPoint)
        else:
            if movements is not None and len(tabuRate) > 0:
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

def onePointParetoSearch(initialState, instance, movementOperator, lexPoints, neiborsGenerated = 10, iterationAmount = 50, maxIterationsWithoutImprovement=5, movementSize = 3, tabuListSize = 20, tabuTenure = 5, alpha = 0.5, maxWorkers = 10):
    # 1. Inicialización y obtencion de parametros
    nonDominatedPoints = []
    aux, solverTime, callIterations, callNodes = calculateFitnessParallel(instance, initialState, max_workers=maxWorkers, alphaValue=alpha, lexPoints=lexPoints)
    nonDominatedPoints.extend(copy.deepcopy(aux))

    #print(f"DEBUG_SOLVER - Elementos válidos retornados por el solver inicial: {len(aux) if aux else 0}")

    foundPoints = []
    foundPoints.extend(copy.deepcopy(nonDominatedPoints))

    i = 0
    iterationwithoutImprovement = 0
    totalSolverIterations = callIterations
    totalSolverNodes = callNodes

    tabu = createTabuList(instance)
    addedTabus = []

    solverTime = 0
    
    stopped = [False, None]

    while i < iterationAmount: 
        #print(f"Iteracion {i+1}/{iterationAmount}")

        neighborhood = []
        neighborMovements = []
        tabuNeighborhood = []

        # 2. Generar vecinos y remover duplicados
        for point in nonDominatedPoints:
            #print (f"Explorando punto: {point.state}, Infra: {point.Infrastructure}, Trans: {point.Transport}, Explorado: {point.explored}")
            if point.explored == False:
                point.explored = True
                neighborhood, neighborMovements, tabuNeighborhood, solverData = hybridNeighborGeneration(point, instance, movementOperator, neiborsGenerated, maxWorkers=maxWorkers,fixingSize=movementSize, 
                                                                                            alphaValue=alpha, lexPoints=lexPoints, tabu=tabu, movementSize=movementSize)
                
                totalSolverIterations += solverData[0]
                totalSolverNodes += solverData[1]
                break

        neighborhood = removeDuplicateStates(neighborhood)
        
        if len(tabuNeighborhood) > 0: 
            tabuNeighborhood = removeDuplicateStates(tabuNeighborhood)

        # 3. Evaluar vecinos marcados como tabu con criterio de aspiración
        if len(tabuNeighborhood) > 0:
            validTabuPoints, time, callIterations, callNodes = AspirationCriteria(instance, tabuNeighborhood, nonDominatedPoints, foundPoints, alphaValue=alpha, lexPoints=lexPoints, maxWorkers=maxWorkers)
            solverTime += time
            totalSolverIterations += callIterations
            totalSolverNodes += callNodes
        else:
            validTabuPoints = []

        notFound, alreadyFound = checkIfFound(neighborhood, foundPoints)

        # 4. Evaluar vecinos no encontrados
        paretoPoints, time, callIterations, callNodes = calculateFitnessParallel(instance, notFound, max_workers=maxWorkers, alphaValue=alpha, lexPoints=lexPoints)
            
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

        if len(allPotentialPoints) > 0:
            nonDominatedPoints, tabuRate = checkDominance(allPotentialPoints, nonDominatedPoints, neighborMovements)

        nonDominatedPoints = removeDuplicatePoints(nonDominatedPoints)

        # Check if any of the NEWLY evaluated points made it into the front
        currentFrontStates = set(p.state for p in nonDominatedPoints)
        # An improvement only counts if a NEW state was added that isn't in the "history"
        actualImprovement = any(p.state in currentFrontStates for p in newlyEvaluatedPoints)
        frontChanged = not currentFrontStates.issubset(previousFrontStates)

        noPointToExplore = True
        for point in nonDominatedPoints:
            if point.explored == False:
                noPointToExplore = False
                break
        
        totalSolverIterations += callIterations
        totalSolverNodes += callNodes

        if noPointToExplore:
            #print("No se han encontrado nuevos puntos para explorar, terminando búsqueda.")
            stopped = [True, i]
            nonDominatedPoints = removeDuplicatePoints(nonDominatedPoints)
            break

        if actualImprovement and frontChanged:
            iterationwithoutImprovement = 0
            #print(f"Iteración {i+1}/{iterationAmount} - Nuevo punto no dominado encontrado! Total en el frente: {len(nonDominatedPoints)}")
        elif iterationwithoutImprovement >= maxIterationsWithoutImprovement:
            #print("No se han encontrado nuevos puntos no dominados en las últimas 5 iteraciones, terminando búsqueda.")
            stopped = [True, i]
            nonDominatedPoints = removeDuplicatePoints(nonDominatedPoints)
            break
        else:
            iterationwithoutImprovement += 1
            i += 1
            #print(f"Iteración {i}/{iterationAmount} - No se encontraron nuevos puntos no dominados. Iteraciones sin mejora: {iterationwithoutImprovement}")
            continue
        
        # 6. Añadir tabu de los movimientos más frecuentes en los nuevos puntos no dominados encontrados
        sortedTabuRate = sorted(tabuRate.items(), key=lambda x: x[1], reverse=True)
        
        tabuMovesToAdd = {}

        for j in range(tabuTenure):
            move = sortedTabuRate[j]
            moveKey = int(move[0][0])
            moveValue = move[1]
            tabuMovesToAdd[moveKey] = moveValue

        tabu, addedTabus = addTabu(tabuMovesToAdd, tabu, addedTabus)

        if len(addedTabus) > tabuListSize:
            tabu, addedTabus = removeLastTabu(tabuTenure, tabu, addedTabus)
        
        foundPoints.extend(copy.deepcopy(paretoPoints))

        i += 1
    
    return nonDominatedPoints, solverTime, stopped, totalSolverIterations, totalSolverNodes