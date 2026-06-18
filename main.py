import matplotlib.pyplot as plt
import sys
import os
from time import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.initialSolution import generateInitialSolution
from src.utils import loadDatInstance, calcularHipervolumen, invertedGenerationalDistance, spacing,exportData, loadConfig, parseNumCds, getReportData
from src.solver import calculateFitnessParallel
from src.plotting import plotParetoFront
from src.TPLS_MPS import MultiPointParetoSearch
from src.TPLS_OPS import onePointParetoSearch
from lexsrc.utils import loadEpsilonResults

#Inicialización de variables globales y constantes
license_UIDD = "8b9ba85b-4781-4c85-94c3-2b6fcb16b02e"

if __name__ == "__main__":
    configData = loadConfig("config.json")
    numExperiments = configData["numExperiments"]
    plsParams = configData["plsParams"]
    instancesList = configData["instances"]

    license_UIDD = "8b9ba85b-4781-4c85-94c3-2b6fcb16b02e"
    dataUrl = ""

    for instanceName in instancesList: 
        # 1. Cargar las intancias (ya sea desde el json o de las urls)
        print(f"###################################")
        print(f"Ejecución de la instancia: {instanceName}")
        print(f"###################################")

        currentInstance = loadDatInstance(instanceName)
            
        numCds = parseNumCds(currentInstance)
        print(f"Instancia cargada con {numCds} CDs candidatos.")

        # 2. Obtener los puntos lexicográficos extremos para cada objetivo
        previousResults = loadEpsilonResults(f"fronts/{instanceName}.json", instanceName)

        # 3. Obtener una solución inicial aleatoria
        initialState = generateInitialSolution(plsParams['operators']["initialization"], currentInstance)
        initialPoints, _, _ = calculateFitnessParallel(currentInstance, initialState, plsParams['solver'], max_workers=plsParams['maxWorkers'], alphaValue=plsParams['alpha'], lexPoints=[previousResults['transMax'], previousResults['infraMax']])
        print (f"solucion inicial: {initialState}")
        # 4. Ejecutar la busqueda local
        experimentRegistry = {
            'executionTime': [],
            'amountCallsSolver': [],
            'amountNodesSolver': [],
            'hypervolume': [],
            'invertedGenerationalDistance': [],
            'spacing': [],
            'points': [],
            'executedIterations': []
        }

        for experiment in range(numExperiments):
            print(f"##############################")
            print(f"Experimento {experiment+1}/{numExperiments} con alpha={plsParams['alpha']}")
            print(f"##############################")

            match plsParams['operators']['searchMethod']:

                case "steepestDescent":
                    timeStart = time()
                    finalParetoFront, solverTime, stopped, solverIterations, solverNodes = MultiPointParetoSearch(
                        initialState=initialState, 
                        instance=currentInstance,
                        movementOperator=plsParams['operators']['neiborhoodGeneration'],
                        solver=plsParams['solver'],
                        lexPoints=[previousResults['transMax'], previousResults['infraMax']], 
                        iterationAmount=plsParams['iterations'], 
                        maxIterationsWithoutImprovement=plsParams['maxIterationsWithoutImprovement'],
                        movementSize=plsParams['movementSize'], 
                        fixingSize=plsParams['fixListSize'],
                        tabuListSize=plsParams['tabuListSize'], 
                        tabuTenure=plsParams['tabuTenure'], 
                        alpha=plsParams['alpha'],
                        maxWorkers=plsParams['maxWorkers']
                    )
                    timeEnd = time()
                case "firstDescent":
                    timeStart = time()
                    finalParetoFront, solverTime, stopped, solverIterations, solverNodes = onePointParetoSearch(
                        initialState=initialState, 
                        instance=currentInstance,
                        movementOperator=plsParams['operators']['neiborhoodGeneration'],
                        solver=plsParams['solver'],
                        lexPoints=[previousResults['transMax'], previousResults['infraMax']], 
                        iterationAmount=plsParams['iterations'], 
                        fixingSize=plsParams['fixListSize'],
                        movementSize=plsParams['movementSize'], 
                        tabuListSize=plsParams['tabuListSize'], 
                        tabuTenure=plsParams['tabuTenure'], 
                        alpha=plsParams['alpha'],
                        maxWorkers=plsParams['maxWorkers'],
                        maxIterationsWithoutImprovement=plsParams['maxIterationsWithoutImprovement'])
                    timeEnd = time()
                case _:
                    print(f"Método de búsqueda '{plsParams['operators']['searchMethod']}' no reconocido.")
            
            # 5. Recolecion de data
            # 5.1 Obtener metricas de medicion
            hvValue = calcularHipervolumen(finalParetoFront, previousResults['transMin'], previousResults['transMax'], previousResults['infraMin'], previousResults['infraMax'])
            igdValue = invertedGenerationalDistance(previousResults['paretoY'], previousResults['paretoX'], finalParetoFront, [previousResults['infraMin'], previousResults['infraMax']], [previousResults['transMin'], previousResults['transMax']])
            spacingValue = spacing(finalParetoFront)
            
            # 5.2 Registrar experimento
            experimentRegistry['executionTime'].append(timeEnd-timeStart)
            experimentRegistry['amountCallsSolver'].append(solverIterations)
            experimentRegistry['amountNodesSolver'].append(solverNodes)
            experimentRegistry['hypervolume'].append(hvValue)
            experimentRegistry['invertedGenerationalDistance'].append(igdValue)
            experimentRegistry['spacing'].append(spacingValue)
            experimentRegistry['points'].append(finalParetoFront)
            experimentRegistry['executedIterations'].append(stopped[1])

        # 5.3 Procesar los datos
        tplsData = getReportData(experimentRegistry)
        tplsData['usedStrategy'] = plsParams['operators']['searchMethod']
        tplsData['amountOfExperiments'] = numExperiments
        tplsData['maxIterations'] = plsParams['iterations']

        # 5.4 Crear el reporte
        exportData(instanceName, currentInstance, numCds, previousResults, tplsData)

        # 6. Plotting y visualización de los resultados
        if finalParetoFront:
            plotParetoFront(tplsData['bestFront'], previousResults, initialPoints, instanceName, plsParams['operators']['searchMethod'])
        else:
            print("No solutions were found to plot.")