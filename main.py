from amplpy import AMPL
from matplotlib.dates import TH
import numpy as np
import requests
import matplotlib.pyplot as plt
import sys
import os
from time import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.initialSolution import generateInitialSolution
from src.utils import loadDatInstance, calcularHipervolumen, invertedGenerationalDistance, spacing,exportData, loadConfig, parseNumCds
from src.solver import calculateFitnessParallel
from src.TPLS_MPS import MultiPointParetoSearch
from src.TPLS_OPS import onePointParetoSearch
from lexsrc.utils import saveEpsilonFront, loadEpsilonResults

if __name__ == "__main__":
    #Carga de configuracion de ejecucion
    configData = loadConfig("config.json")
    numExperiments = configData["numExperiments"]
    plsParams = configData["plsParams"]
    instancesList = configData["instances"]

    license_UIDD = "8b9ba85b-4781-4c85-94c3-2b6fcb16b02e"
    dataUrl = ""

    getEpsilon = True

    for instanceName in instancesList: 
        # 1. Cargar las intancias
        print(f"Ejecución de la instancia: {instanceName}")

        currentInstance = loadDatInstance(instanceName)
            
        numCds = parseNumCds(currentInstance)
        print(f"Instancia cargada con {numCds} CDs candidatos.")

        # 2. Cargar el frente de referencia y resultados previos de la instacia.
        previousResults = loadEpsilonResults(f"fronts/{instanceName}.json", instanceName)

        # 3. Obtener una solución inicial aleatoria
        initialState = generateInitialSolution(plsParams['operators']["initialization"], currentInstance)
        initialPoints, _, _, _ = calculateFitnessParallel(currentInstance, initialState, max_workers=plsParams['maxWorkers'], alphaValue=plsParams['alpha'], lexPoints=[previousResults['transMax'], previousResults['infraMax']])
        print (f"solucion inicial: {initialState}")

        # 4. Ejecutar la busqueda local
        experimentRegistry = []

        for experiment in range(numExperiments):
            print(f"Experimento {experiment+1}/{numExperiments} con alpha={plsParams['alpha']}")
            
            timeZero = time()

            match plsParams['operators']['searchMethod']:
                case "steepestDescent":
                    timeStart = time()
                    finalParetoFront, solverTime, stopped, solverIterations, solverNodes = MultiPointParetoSearch(
                        initialState, 
                        currentInstance,
                        plsParams['operators']['neiborhoodGeneration'],
                        [previousResults['transMax'], previousResults['infraMax']], 
                        iterationAmount=plsParams['iterations'], 
                        neiborsGenerated=plsParams['neiborsGenerated'],
                        maxIterationsWithoutImprovement=plsParams['maxIterationsWithoutImprovement'],
                        movementSize=plsParams['movementSize'], 
                        tabuListSize=plsParams['tabuListSize'], 
                        tabuTenure=plsParams['tabuTenure'], 
                        alpha=plsParams['alpha'],
                        maxWorkers=plsParams['maxWorkers']
                    )
                    timeEnd = time()
                case "firstDescent":
                    timeStart = time()
                    finalParetoFront, solverTime, stopped, solverIterations, solverNodes = onePointParetoSearch(
                        initialState, 
                        currentInstance,
                        plsParams['operators']['neiborhoodGeneration'],
                        [previousResults['transMax'], previousResults['infraMax']], 
                        iterationAmount=plsParams['iterations'], 
                        neiborsGenerated=plsParams['neiborsGenerated'],
                        movementSize=plsParams['movementSize'], 
                        tabuListSize=plsParams['tabuListSize'], 
                        tabuTenure=plsParams['tabuTenure'], 
                        alpha=plsParams['alpha'],
                        maxWorkers=plsParams['maxWorkers'],
                        maxIterationsWithoutImprovement=plsParams['maxIterationsWithoutImprovement'])
                    timeEnd = time()
                case _:
                    print(f"Método de búsqueda '{plsParams['operators']['searchMethod']}' no reconocido.")

            # 5. Recopilacion de datos de la ejecucion
            if getEpsilon:
                hvValue = calcularHipervolumen(finalParetoFront, previousResults['transMin'], previousResults['transMax'], previousResults['infraMin'], previousResults['infraMax'])
                igdValue = invertedGenerationalDistance(previousResults['paretoX'], previousResults['paretoY'], finalParetoFront, [previousResults['infraMin'], previousResults['infraMax']], [previousResults['transMin'], previousResults['transMax']])
                spacingValue = spacing(finalParetoFront)

            if getEpsilon:
                tplsInfo = {
                'executionTime': timeEnd - timeStart,
                'hypervolume': hvValue,
                'invertedGenerationalDistance': igdValue,
                'spacing': spacingValue,
                'points': finalParetoFront,
                'usedStrategy': plsParams['operators']['searchMethod'],
                'totalSolverIterations': solverIterations,
                'totalSolverNodes': solverNodes
                }
            else:
                tplsInfo = {
                'executionTime': timeEnd - timeStart,
                'points': finalParetoFront,
                'usedStrategy': plsParams['operators']['searchMethod']
                }

            print("######################################")
            print(f"Total de iteraciones = {tplsInfo['totalSolverIterations']} y total de nodos = {tplsInfo['totalSolverNodes']}")
            print("######################################")

            if stopped[0]:
                tplsInfo['stopped'] = True
                tplsInfo['stoppingIteration'] = stopped[1]
                tplsInfo['amountIterations'] = iterationAmount=plsParams['iterations']
            else:
                tplsInfo['stopped'] = False
                tplsInfo['amountIterations'] = iterationAmount=plsParams['iterations']

        exportData(instanceName, currentInstance, numCds, previousResults, getEpsilon, tplsInfo)

        # 6. Plotting y visualización de los resultados
        if finalParetoFront:
        # Extract objective values from the paretoPoint instances 
        # objValueX = Infrastructure Cost, objValueY = Transport Cost
            infra_costs = [p.Infrastructure for p in finalParetoFront]
            trans_costs = [p.Transport for p in finalParetoFront]

            # Create the plot
            plt.figure(figsize=(10, 6))
            
            # Plot the lexicographic points for reference
            # In main.py, change the order to (Infra, Transport)
            if getEpsilon:
                plt.scatter([previousResults['transMin'], previousResults['transMax']], [previousResults['infraMax'], previousResults['infraMin']], c=['blue', 'red'])
                plt.plot(previousResults['paretoX'], previousResults['paretoY'], marker='o', linestyle='-', color='green', label='Epsilon Frontier') 

            # Plot individual points
            plt.scatter(trans_costs, infra_costs, color='purple', zorder=5, label='Pareto Optimal Points')

            # Optional: Draw a line connecting the points to visualize the 'Frontier'
            # We sort by Infrastructure Cost to ensure the line connects points in order
            sorted_front = sorted(finalParetoFront, key=lambda p: p.Infrastructure)
            x_line = [p.Infrastructure for p in sorted_front]
            y_line = [p.Transport for p in sorted_front]
            plt.plot(y_line, x_line, color='blue', linestyle='--', alpha=0.6, label='Pareto Frontier')

            # Highlight the initial solution
            initialInfra = [p.Infrastructure for p in initialPoints]
            initialTrans = [p.Transport for p in initialPoints]
            plt.scatter(initialTrans, initialInfra, color='orange', marker='X', s=100, label='Initial Solution')

            # Labels and Titles
            plt.title(f"Results for {instanceName}")
            plt.xlabel('Transport Cost ($)')
            plt.ylabel('Infrastructure Cost ($)')
            plt.grid(True, linestyle=':', alpha=0.7)
            plt.legend()

            # Save the plot
            plt.savefig(f"{instanceName}_using_{plsParams['operators']['searchMethod']}_results.png")
            #plt.show()
            print(f"Visualisation saved as '{instanceName}_using_{plsParams['operators']['searchMethod']}_results.png'")
        else:
            print("No solutions were found to plot.")