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
from src.utils import loadJsonInstance, loadTextInstance, printSummary, randomSolution, calcularHipervolumen
from src.TPLS import tabuLocalParetoSearch
from lexsrc.model import instanceToAmpl, mTransport, mInfrastructure
from lexsrc.solver import solveInstance, solveEpsilon

instancesChatGPT = [
    "https://gist.githubusercontent.com/athersoft/c6baed29465f509c315c2f5fa7db93b4/raw/393643d794fba167cfe9ec6a3c5c91c3a8cd1d48/80x40-chatgpt.dat",
    "https://gist.githubusercontent.com/athersoft/60304f8af5b3dfc33cf62094a0cc78d6/raw/0a1eb0599c84622853b9294abe337df32b60df7a/100x50-chatgpt.dat",
    "https://gist.githubusercontent.com/athersoft/6bd1bee9640084322d0f19f1764e4124/raw/57c3d1ee8f1c47acbdcbab47cffb826e7d5eea9a/120x60-chatgpt",
    "https://gist.githubusercontent.com/athersoft/00335c96a7ff7e52a013910b5d657091/raw/89959b566cb1f26909ba3d7d38777d4a4188ee2d/140x70-chatgpt.dat",
    "https://gist.github.com/athersoft/f45eca17f1a5f696d11725d08ed4fdaf/raw/89ab7952b5fc947e3ed6aa4326420658b7dddd33/200x100-chatgpt"
]

instancesGrok = [
    "https://gist.githubusercontent.com/athersoft/3e4fdb3ee806d5cca5c2c1952e1de007/raw/acc503904e7972867591104ff55240c6ddb2dcdb/80x40-grok.dat",
    "https://gist.githubusercontent.com/athersoft/39e316457aa8b8eb03b51ebae423f316/raw/97ea051b3c129ff0c6aeffad92ef8da34cd5693f/100x50-grok.dat",
    "https://gist.githubusercontent.com/athersoft/6daf6a7b4ac2062662601f83e1d2d2bd/raw/e0c5824a7b6cc59734142e8ca66542311b0b1c01/120x60-grok.dat",
    "https://gist.githubusercontent.com/athersoft/9e83d347c7c6516779da64f686573c18/raw/7466c7a863bdfbb6e8ec66a3e730d45b9a84774a/140x70-grok.dat"
]

instancesGemini = [
    "https://gist.github.com/athersoft/e0bfbcdc2bf4beda0ba81daeb87b8a2d/raw/ec7e78f81a177649f83e1fead4054adab51357a0/80x40-gemini.dat",
    "https://gist.github.com/athersoft/b3ce8c66ce3c51e174d81a7ca9eaefd9/raw/916f23718c276601df5a0631c1c59421470767b9/100x50-gemini.dat",
    "https://gist.github.com/athersoft/3853f927779746cb3b8fb8650b8ff4d3/raw/c17057b3fbef7772a522a4891f1fc4be22ff2884/120x60-gemini.dat",
    "https://gist.github.com/athersoft/1b2d3540308e38df4cd8cbaf28348593/raw/db7d39e493d701d12cfe42565152dc01057308bd/140x70-gemini.dat",
    "https://gist.github.com/athersoft/1a26d2dfe533bf2b31fcda682d1b82e7/raw/435db55d97f212e22d8a82bf1d4de3afec6fcd14/100x200-gemini.dat"
]

instancesDeepseek = [
    "https://gist.github.com/athersoft/da76049ae985f515cf3b9759083d6f6d/raw/b85800b5ff3c5a9754cd6af95113e49d2b6c98b9/80x40-deepseek.dat",
    "https://gist.github.com/athersoft/383e7ddf48dcf0d51af9ab5bec757eae/raw/a59a7a3fd7a7cc5b1b610a6b8a9cc5384928e791/100x50-deepseek.dat",
    "https://gist.github.com/athersoft/5544cef94c9382246010c575ff64e8d1/raw/dabe96f65671e4ac6d278d8fe6fe1f51c822b10d/120x60-deepseek.dat",
    "https://gist.github.com/athersoft/63415c7f2205b1c61129ebc1ee3cfcd8/raw/e1635522259d8076bad99dcc0865d11de6b01c96/140x70-deepseek.dat"
]

license_UIDD = "8b9ba85b-4781-4c85-94c3-2b6fcb16b02e"

if __name__ == "__main__":
    currentUrl = instancesGemini[2]
    currentInstance = ""

    print(f"Descargando archivo .dat desde Gist...")
    try:
        response = requests.get(currentUrl)
        response.raise_for_status()
        currentInstance = response.text

    except Exception as e:
        print(f"Error descargando: {e}")

    # 1. Cargar las intancias (ya sea desde el json o de las urls)
    if 1==2:
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
    transportMin, aux = solveInstance(currentInstance, mTransport)
    aux, infraMax = solveInstance(currentInstance, mInfrastructure, transportMin)
    
    aux, infraMin = solveInstance(currentInstance, mInfrastructure)
    transportMax, aux = solveInstance(currentInstance, mTransport, infraMin)

    print(f"Punto X lexicográfico de infraestructura e inventario {transportMax}")
    print(f"Punto Y lexicográfico de infraestructura e inventario {infraMin}")
    print(f"Punto X lexicográfico de transporte: {transportMin}")
    print(f"Punto Y lexicográfico de transporte: {infraMax}")

    steps = 10
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
    
    # 4. Ejecutar la busqueda local
    alpha = 0.9
    while True:
        if alpha == 1:
            break
        if 1==1:
            time0 = time()
            finalParetoFront, solverTime = tabuLocalParetoSearch(cdList, clientList, K, TH, 50, 3, int(len(cdList)/2), int(len(cdList)/4), alpha)
            time1 = time()
            executionTime = time1 - time0
        else:
            finalParetoFront = paretoLocalSearch(cdList, clientList, K, TH, 50)

        # 5. Puntos no dominados encontrados y tiempo de ejecución
        for point in finalParetoFront:
            print(f"state {point.state}, Infra: {point.objValueX}, Transport: {point.objValueY}")
        print(f"Total execution time: {executionTime:.2f} seconds (Solver time: {solverTime:.2f} seconds)")

        # 6. Plotting y visualización de los resultados
        if finalParetoFront:
        # Extract objective values from the paretoPoint instances 
        # objValueX = Infrastructure Cost, objValueY = Transport Cost
            infra_costs = [p.objValueX for p in finalParetoFront]
            trans_costs = [p.objValueY for p in finalParetoFront]

            # Create the plot
            plt.figure(figsize=(10, 6))
            
            # Plot individual points
            plt.scatter(infra_costs, trans_costs, color='red', zorder=5, label='Pareto Optimal Points')

            # Optional: Draw a line connecting the points to visualize the 'Frontier'
            # We sort by Infrastructure Cost to ensure the line connects points in order
            sorted_front = sorted(finalParetoFront, key=lambda p: p.objValueX)
            x_line = [p.objValueX for p in sorted_front]
            y_line = [p.objValueY for p in sorted_front]
            plt.plot(x_line, y_line, color='blue', linestyle='--', alpha=0.6, label='Pareto Frontier')

            # Plot the lexicographic points for reference
            plt.plot(paretoX, paretoY, marker='o', linestyle='-', color='green')
            plt.scatter([transportMin, transportMax], [infraMax, infraMin], c=['blue', 'red'], zorder=5)


            # Labels and Titles
            plt.title('Trade-off between Infrastructure and Transport Costs')
            plt.xlabel('Infrastructure Cost ($)')
            plt.ylabel('Transport Cost ($)')
            plt.grid(True, linestyle=':', alpha=0.7)
            plt.legend()
            plt.show()

            # Save the plot
            plt.savefig('pareto_front_results.png')
            print("Visualisation saved as 'pareto_front_results.png'")
        else:
            print("No solutions were found to plot.")
        
        alpha = 1
        