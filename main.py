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
from lexsrc.model import mTransport, mInfrastructure
from lexsrc.solver import solveInstance, solveEpsilon, filterEpsilonFront
from lexsrc.utils import saveEpsilonFront, loadEpsilonResults

instancesChatGPT = [
    "https://gist.githubusercontent.com/athersoft/c6baed29465f509c315c2f5fa7db93b4/raw/0934f2edc08cd275e8ac98872e6e1d4cc13cb003/80x40-chatgpt.dat",
    "https://gist.githubusercontent.com/athersoft/60304f8af5b3dfc33cf62094a0cc78d6/raw/c90d6a6a62cb5313862852bc321267595ec10caf/100x50-chatgpt.dat",
    "https://gist.githubusercontent.com/athersoft/6bd1bee9640084322d0f19f1764e4124/raw/fd4b2123efbb389f476ff5f2dc8bdf5583d92fdd/120x60-chatgpt.dat",
    "https://gist.githubusercontent.com/athersoft/00335c96a7ff7e52a013910b5d657091/raw/ab6c7eb89824c03233493c0b5a4eb95d721062c5/140x70-chatgpt.dat",
    "https://gist.githubusercontent.com/athersoft/f45eca17f1a5f696d11725d08ed4fdaf/raw/94c125e1f576214c16e04a216d00b153c919584b/200x100-chatgpt"
]

instancesGrok = [
    "https://gist.githubusercontent.com/athersoft/3e4fdb3ee806d5cca5c2c1952e1de007/raw/acc503904e7972867591104ff55240c6ddb2dcdb/80x40-grok",
    "https://gist.githubusercontent.com/athersoft/39e316457aa8b8eb03b51ebae423f316/raw/97ea051b3c129ff0c6aeffad92ef8da34cd5693f/100x50-grok.dat",
    "https://gist.githubusercontent.com/athersoft/6daf6a7b4ac2062662601f83e1d2d2bd/raw/96468a3c756cb9d578e5e68c8b6c62d88c9addac/120x60-grok.dat",
    "https://gist.githubusercontent.com/athersoft/9e83d347c7c6516779da64f686573c18/raw/01f61ef6960ee5ab50d464aeb9a7e90b80a4227f/140x70-grok.dat"
]

instancesGemini = [
    "https://gist.github.com/athersoft/e0bfbcdc2bf4beda0ba81daeb87b8a2d/raw/ec7e78f81a177649f83e1fead4054adab51357a0/80x40-gemini.dat",
    "https://gist.github.com/athersoft/b3ce8c66ce3c51e174d81a7ca9eaefd9/raw/916f23718c276601df5a0631c1c59421470767b9/100x50-gemini.dat",
    "https://gist.githubusercontent.com/athersoft/3853f927779746cb3b8fb8650b8ff4d3/raw/c49a3b1d259d99d38b8c14daa33c6672d403924b/120x60-gemini.dat",
    "https://gist.githubusercontent.com/athersoft/1b2d3540308e38df4cd8cbaf28348593/raw/2f111b387f7d51e3d850ac1905e8d6f72b07d4b3/140x70-gemini.dat",
    "https://gist.github.com/athersoft/1a26d2dfe533bf2b31fcda682d1b82e7/raw/435db55d97f212e22d8a82bf1d4de3afec6fcd14/100x200-gemini.dat"
]

instancesDeepseek = [
    "https://gist.github.com/athersoft/da76049ae985f515cf3b9759083d6f6d/raw/b85800b5ff3c5a9754cd6af95113e49d2b6c98b9/80x40-deepseek.dat",
    "https://gist.github.com/athersoft/383e7ddf48dcf0d51af9ab5bec757eae/raw/a59a7a3fd7a7cc5b1b610a6b8a9cc5384928e791/100x50-deepseek.dat",
    "https://gist.github.com/athersoft/5544cef94c9382246010c575ff64e8d1/raw/dabe96f65671e4ac6d278d8fe6fe1f51c822b10d/120x60-deepseek.dat",
    "https://gist.github.com/athersoft/63415c7f2205b1c61129ebc1ee3cfcd8/raw/e1635522259d8076bad99dcc0865d11de6b01c96/140x70-deepseek.dat"
]

instanciaPaper = ["https://gist.githubusercontent.com/athersoft/2dcb176d505a41cffdbcc568682576b5/raw/ac9331d7f6fcecf3fa9b97ca41b0e9d6b1f0b889/instanciaPaper"]

instancesSpecial = [
    "https://gist.githubusercontent.com/athersoft/ae222648b85aa417c53a841a3e39eac7/raw/6afa7e71baf4a75951bc1dab042cf89ce6ffbc4e/inventarioAbsurdo",
    "https://gist.githubusercontent.com/athersoft/61aa11e8d3cef6584417439e5fcc4808/raw/bc28fbcea9b088c0abec974fa5537a6e02d9da98/infraestructuraProhibitiva",
    "https://gist.githubusercontent.com/athersoft/d3a58f54fc61ad124e75884ae5595a32/raw/5fcbb5c1eb832f3d1ffee9fb85280e196f3da5b4/demandaExtrema",
    "https://gist.githubusercontent.com/athersoft/87326bd12029819eb826b9cd3db07808/raw/86c4880ef37dc7e11d2049ec5ccfdb85dfb8c650/capacidadRestringida",
    "https://gist.githubusercontent.com/athersoft/259c57976bd4ff835394be1b0f91aae0/raw/f10a2f6c1ba13520ecfebceab8478f03178e08b8/altaDispersion"
]

topologicos = [
    "https://gist.githubusercontent.com/athersoft/bf02498ff184b433148c77bdf18f8960/raw/10cf7213cb17f488affe614031ff4494892cf350/15x30_topologico",
    "https://gist.githubusercontent.com/athersoft/118f95592d497f953e4f8ef3ee8b9d8b/raw/f734c167ba3d9ddbe6a54c3c7add84d93508b96e/25x50_topologico",
    "https://gist.githubusercontent.com/athersoft/b6eb4abb0ea718ea6ad153d376764e32/raw/d7fea0bbb8efc61f8b533262c2175c5991a10cf5/40x80_topologico",
    "https://gist.githubusercontent.com/athersoft/b738c1d009151e4c881beca1a78bbcdb/raw/d46c9ab38337d3618b1e67eb8a2868c12592d3e3/50x100_topologico"
]

instanciasParaElPaper = [
    "https://gist.githubusercontent.com/athersoft/118f95592d497f953e4f8ef3ee8b9d8b/raw/f734c167ba3d9ddbe6a54c3c7add84d93508b96e/25x50_topologico",
    "https://gist.githubusercontent.com/athersoft/bf02498ff184b433148c77bdf18f8960/raw/10cf7213cb17f488affe614031ff4494892cf350/15x30_topologico",
    "https://gist.githubusercontent.com/athersoft/61aa11e8d3cef6584417439e5fcc4808/raw/bc28fbcea9b088c0abec974fa5537a6e02d9da98/infraestructuraProhibitiva",
    "https://gist.githubusercontent.com/athersoft/3853f927779746cb3b8fb8650b8ff4d3/raw/c49a3b1d259d99d38b8c14daa33c6672d403924b/120x60-gemini.dat"
]

instanciasParaLaPresentacion = [
    "https://gist.githubusercontent.com/athersoft/3e4fdb3ee806d5cca5c2c1952e1de007/raw/acc503904e7972867591104ff55240c6ddb2dcdb/80x40-grok",
    "https://gist.githubusercontent.com/athersoft/39e316457aa8b8eb03b51ebae423f316/raw/97ea051b3c129ff0c6aeffad92ef8da34cd5693f/100x50-grok.dat",
    "https://gist.githubusercontent.com/athersoft/6daf6a7b4ac2062662601f83e1d2d2bd/raw/96468a3c756cb9d578e5e68c8b6c62d88c9addac/120x60-grok.dat",
    "https://gist.githubusercontent.com/athersoft/118f95592d497f953e4f8ef3ee8b9d8b/raw/f734c167ba3d9ddbe6a54c3c7add84d93508b96e/25x50_topologico",
    "https://gist.githubusercontent.com/athersoft/bf02498ff184b433148c77bdf18f8960/raw/10cf7213cb17f488affe614031ff4494892cf350/15x30_topologico",
    "https://gist.github.com/athersoft/e0bfbcdc2bf4beda0ba81daeb87b8a2d/raw/ec7e78f81a177649f83e1fead4054adab51357a0/80x40-gemini.dat",
    "https://gist.github.com/athersoft/b3ce8c66ce3c51e174d81a7ca9eaefd9/raw/916f23718c276601df5a0631c1c59421470767b9/100x50-gemini.dat",
    "https://gist.githubusercontent.com/athersoft/3853f927779746cb3b8fb8650b8ff4d3/raw/c49a3b1d259d99d38b8c14daa33c6672d403924b/120x60-gemini.dat"

]

test = ["https://gist.githubusercontent.com/athersoft/118f95592d497f953e4f8ef3ee8b9d8b/raw/f734c167ba3d9ddbe6a54c3c7add84d93508b96e/25x50_topologico"]

#Inicialización de variables globales y constantes
license_UIDD = "8b9ba85b-4781-4c85-94c3-2b6fcb16b02e"

if __name__ == "__main__":
    configData = loadConfig("config.json")
    numExperiments = configData["numExperiments"]
    plsParams = configData["plsParams"]
    instancesList = configData["instances"]

    license_UIDD = "8b9ba85b-4781-4c85-94c3-2b6fcb16b02e"
    dataUrl = ""

    getEpsilon = False

    for instanceName in instancesList: 
        # 1. Cargar las intancias (ya sea desde el json o de las urls)
        print(f"Ejecución de la instancia: {instanceName}")

        currentInstance = loadDatInstance(instanceName)
            
        numCds = parseNumCds(currentInstance)
        print(f"Instancia cargada con {numCds} CDs candidatos.")

        # 2. Obtener los puntos lexicográficos extremos para cada objetivo
        previousResults = loadEpsilonResults(f"fronts/{instanceName}.json", instanceName)

        transportMin = previousResults['transMin']
        transportMax = previousResults['transMax']
        infraMin = previousResults['infraMin']
        infraMax = previousResults['infraMax']
        paretoX = previousResults['paretoX']
        paretoY = previousResults['paretoY']

        # 3. Obtener una solución inicial aleatoria
        initialState = generateInitialSolution(plsParams['operators']["initialization"], currentInstance)
        initialPoints, _ = calculateFitnessParallel(currentInstance, initialState, max_workers=plsParams['maxWorkers'], alphaValue=plsParams['alpha'], lexPoints=[transportMax, infraMax])
        print (f"solucion inicial: {initialState}")
        # 4. Ejecutar la busqueda local
        experimentRegistry = []
        iteration = 0

        for experiment in range(numExperiments):
            print(f"Experimento {experiment+1}/{numExperiments} con alpha={plsParams['alpha']}")
            
            timeZero = time()

            if plsParams['operators']['searchMethod'] == "multiPoint":
                timeStart = time()
                finalParetoFront, solverTime, stopped = MultiPointParetoSearch(
                    initialState, 
                    currentInstance,
                    plsParams['operators']['neiborhoodGeneration'],
                    [transportMax, infraMax], 
                    iterationAmount=plsParams['iterations'], 
                    maxIterationsWithoutImprovement=plsParams['maxIterationsWithoutImprovement'],
                    movementSize=plsParams['movementSize'], 
                    tabuTenure=plsParams['tabuListSize'], 
                    amountToAdd=plsParams['tabuTenure'], 
                    alpha=plsParams['alpha'],
                    maxWorkers=plsParams['maxWorkers']
                )
                timeEnd = time()
            else:
                timeStart = time()
                finalParetoFront, solverTime, stopped = onePointParetoSearch(
                    initialState, 
                    currentInstance,
                    plsParams['operators']['neiborhoodGeneration']
                    [transportMax, infraMax], 
                    iterationAmount=plsParams['iterations'], 
                    movementSize=plsParams['movementSize'], 
                    tabuListSize=plsParams['tabuListSize'], 
                    tabuTenure=plsParams['tabuTenure'], 
                    alpha=plsParams['alpha'],
                    maxWorkers=plsParams['maxWorkers'],
                    maxIterationsWithoutImprovement=plsParams['maxIterationsWithoutImprovement'])
                timeEnd = time()

            # 5. Calculate Hypervolume for this instance
            # Convert objects to (x, y) tuples for your HV function
            if getEpsilon:
                points = [(p.Transport, p.Infrastructure) for p in finalParetoFront]
                hvValue = calcularHipervolumen(points, transportMin, transportMax, infraMin, infraMax)
                igdValue = invertedGenerationalDistance(, finalParetoFront, [infraMin, infraMax], [transportMin, transportMax])

            if getEpsilon:
                tplsInfo = {
                'executionTime': timeEnd - timeStart,
                'hypervolume': hvValue,
                'points': finalParetoFront,
                'usedStrategy': plsParams['operators']['searchMethod']
                }
            else:
                tplsInfo = {
                'executionTime': timeEnd - timeStart,
                'points': finalParetoFront,
                'usedStrategy': plsParams['operators']['searchMethod']
                }

            if stopped[0]:
                tplsInfo['stopped'] = True
                tplsInfo['stoppingIteration'] = stopped[1]
                tplsInfo['amountIterations'] = iterationAmount=plsParams['iterations']
            else:
                tplsInfo['stopped'] = False
                tplsInfo['amountIterations'] = iterationAmount=plsParams['iterations']

        exportData(instanceName, currentInstance, numCds, None, getEpsilon, tplsInfo)

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
                plt.scatter([transportMin, transportMax], [infraMax, infraMin], c=['blue', 'red'])
                plt.plot(paretoX, paretoY, marker='o', linestyle='-', color='green', label='Epsilon Frontier') 

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