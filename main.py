from matplotlib.dates import TH
import numpy as np
import requests
import matplotlib.pyplot as plt
import sys
import os
from time import time
from collections import deque

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from amplpy import AMPL, ampl_notebook
from src.PLS import paretoLocalSearch
from src.utils import getStateTuple, loadJsonInstance, loadTextInstance, printSummary, randomSolution, calcularHipervolumen, aggregateResults, formatExperimentOutput, exportResults
from src.TPLS import tabuLocalParetoSearch
from lexsrc.model import instanceToAmpl, mTransport, mInfrastructure
from lexsrc.solver import solveInstance, solveEpsilon

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

instanciaPaper = "https://gist.githubusercontent.com/athersoft/2dcb176d505a41cffdbcc568682576b5/raw/ac9331d7f6fcecf3fa9b97ca41b0e9d6b1f0b889/instanciaPaper"

#Inicialización de variables globales y constantes
license_UIDD = "8b9ba85b-4781-4c85-94c3-2b6fcb16b02e"
experimentAmount = 1
alpha = 0.5

instances = [instancesGrok, instancesChatGPT, instancesGemini]
instancesNames = ["Grok", "ChatGPT", "Gemini"]

if __name__ == "__main__":
    for i in range(len(instances)):
        instanceSet = instances[i]
        name = instancesNames[i]
        
        j = 1
        for currentUrl in instanceSet: #
            currentInstance = ""

            fileName = f"{name}_{j}"

            print(f"Descargando archivo .dat desde Gist...")
            try:
                response = requests.get(currentUrl)
                response.raise_for_status()
                currentInstance = response.text

            except Exception as e:
                print(f"Error descargando: {e}")

            # 1. Cargar las intancias (ya sea desde el json o de las urls)
            if 1==1:
                if currentInstance:
                    cdList, clientList, K, TH = loadTextInstance(currentInstance)
                    printSummary(cdList, clientList, K, TH)
            else:
                cdList, clientList = loadJsonInstance("small2")
                K = 1.28
                TH = 1
                printSummary(cdList, clientList, K, TH)
                currentInstance = instanceToAmpl(cdList, clientList, K, TH)

            # 2. Obtener los puntos lexicográficos extremos para cada objetivo
            getEpsilon = True
            if getEpsilon:
                transportMin, aux = solveInstance(currentInstance, mTransport)
                aux, infraMax = solveInstance(currentInstance, mInfrastructure, transportMin)
                
                aux, infraMin = solveInstance(currentInstance, mInfrastructure)
                transportMax, aux = solveInstance(currentInstance, mTransport, infraMin)

                print(f"Punto X lexicográfico de infraestructura e inventario {transportMax}")
                print(f"Punto Y lexicográfico de infraestructura e inventario {infraMin}")
                print(f"Punto X lexicográfico de transporte: {transportMin}")
                print(f"Punto Y lexicográfico de transporte: {infraMax}")

                steps = 20
                epsilonSteps = np.linspace(infraMin, infraMax, steps)
                print(epsilonSteps)

                paretoX = []
                paretoY = []

                for step in epsilonSteps:
                    transportCost, infraCost = solveEpsilon(currentInstance, mTransport, step)
                    if transportCost is not None:
                        paretoX.append(transportCost)
                        paretoY.append(infraCost)

            # 3. Obtener una solución inicial aleatoria
            randomSolution(cdList, clientList)
            initialState = getStateTuple(cdList)
            
            # 4. Ejecutar la busqueda local
            experimentRegistry = []
            iteration = 0

            while iteration < experimentAmount:
                if alpha == 1:
                    break
                if 1==1:
                    timeStart = time()
                    finalParetoFront, solverTime = tabuLocalParetoSearch(cdList, clientList, K, TH, 50, 5, int(len(cdList)/2), int(len(cdList)/4), alpha)
                    timeEnd = time()
                    executionTime = timeEnd - timeStart
                else:
                    finalParetoFront = paretoLocalSearch(cdList, clientList, K, TH, 50)
                
                # 5. Calculate Hypervolume for this instance
                # Convert objects to (x, y) tuples for your HV function
                hvPoints = [(p.objValueX, p.objValueY) for p in finalParetoFront]
                hvValue = calcularHipervolumen(hvPoints, infraMax * 1.1, transportMax * 1.1)

                instanceData = {
                        "instanceUrl": currentUrl,
                        "infraMin": infraMin,
                        "infraMax": infraMax,
                        "transportMin": transportMin,
                        "transportMax": transportMax,
                        "initialPointState": initialState,
                        "finalPoints": [(p.objValueX, p.objValueY, p.state) for p in finalParetoFront],
                        "hypervolume": hvValue,
                        "executionTime": timeEnd - timeStart
                    }
                
                experimentRegistry.append(instanceData)
                formatExperimentOutput(currentUrl.split('/')[-1], instanceData)

                iteration += 1
            
            summary = aggregateResults(experimentRegistry)
            print("\n" + "="*40)
            print("GLOBAL EXPERIMENT SUMMARY")
            print("="*40)
            print(f"Instances Processed: {summary['totalInstancesTested']}")
            print(f"Avg Hypervolume:    {summary['averageHypervolume']:.2f}")
            print(f"Avg Execution Time: {summary['averageExecutionTime']:.2f}s")
            
            exportResults(experimentRegistry, summary, f"{fileName}_results.json")

            # 6. Plotting y visualización de los resultados
            if finalParetoFront:
            # Extract objective values from the paretoPoint instances 
            # objValueX = Infrastructure Cost, objValueY = Transport Cost
                infra_costs = [p.objValueX for p in finalParetoFront]
                trans_costs = [p.objValueY for p in finalParetoFront]

                # Create the plot
                plt.figure(figsize=(10, 6))
                
                # Plot the lexicographic points for reference
                # In main.py, change the order to (Infra, Transport)
                if getEpsilon:
                    plt.scatter([infraMax, infraMin], [transportMin, transportMax], c=['blue', 'red'])
                    plt.plot(paretoY, paretoX, marker='o', linestyle='-', color='green', label='Lexicographic') 

                # Plot individual points
                plt.scatter(infra_costs, trans_costs, color='purple', zorder=5, label='Pareto Optimal Points')

                # Optional: Draw a line connecting the points to visualize the 'Frontier'
                # We sort by Infrastructure Cost to ensure the line connects points in order
                sorted_front = sorted(finalParetoFront, key=lambda p: p.objValueX)
                x_line = [p.objValueX for p in sorted_front]
                y_line = [p.objValueY for p in sorted_front]
                plt.plot(x_line, y_line, color='blue', linestyle='--', alpha=0.6, label='Pareto Frontier')

                # Labels and Titles
                plt.title(f"Results for {fileName}")
                plt.xlabel('Infrastructure Cost ($)')
                plt.ylabel('Transport Cost ($)')
                plt.grid(True, linestyle=':', alpha=0.7)
                plt.legend()

                # Save the plot
                plt.savefig(f"{fileName}_results.png")
                plt.show()
                print(f"Visualisation saved as '{fileName}_results.png'")
            else:
                print("No solutions were found to plot.")
            
            j += 1