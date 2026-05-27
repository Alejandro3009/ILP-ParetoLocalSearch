import re
import ast
from amplpy import AMPL
import json
import os
import numpy as np
import time
from collections import deque
from src.model import cd, client, modelo

def printSummary(cds, clients, K, TH):
    print("\n" + "="*55)
    print("              DATA INITIALIZATION SUMMARY")
    print("="*55)
    print(f"GLOBAL PARAMS: K = {K}, TH = {TH}")
    print(f"OBJECT COUNTS: {len(cds)} CDs, {len(clients)} Clients")

    if clients:
        cl = clients[0]
        tc_sample = cl.transportCost[:5]
        print(f"\nSAMPLE CLIENT (ID 0):")
        print(f"  - Demand: {cl.demand} | Variance: {cl.variance}")
        print(f"  - TC Array Sample (CDs 0-4): {tc_sample}")

        if any(cost > 0 for cost in cl.transportCost):
            min_c = min(cl.transportCost)
            best_cd = cl.transportCost.index(min_c)
            print(f"  - Result: SUCCESS! Triplets parsed correctly.")
            print(f"  - Client 0 Best CD: {best_cd} (Cost: {min_c})")
        else:
            print("  - WARNING: TC values are still 0.0. Triplets not found.")
    print("="*55 + "\n")

def getStateTuple(cdList):
    aux = list(c.open for c in cdList)
    for i in range(len(aux)):
        if aux[i] == True:
            aux[i] = 1
        else:
            aux[i] = 0
    return tuple(aux)
    
def getTotalDemand(instanceContent):
    tempAmpl = AMPL()
    tempAmpl.eval("reset;")
    tempAmpl.eval(modelo)
    tempAmpl.eval(instanceContent)

    demand = tempAmpl.getParameter("d").getValues().toDict()
    variance = tempAmpl.getParameter("u").getValues().toDict()

    tempAmpl.close()

    totalDemand = 0
    totalVariance = 0
    for i in range(len(demand)):
        totalDemand += demand[i]
        totalVariance += variance[i]
    return totalDemand + totalVariance

def parseNumCds(ampl_data):
    match = re.search(r'set\s+I\s*:=\s*(.*?);', ampl_data, re.DOTALL | re.IGNORECASE)
    if match:
        content = match.group(1)
        cdList_ids = content.split()
        return len(cdList_ids)
    return 0

def calcularHipervolumen(puntos, minX, maxX, minY, maxY):
    if len(puntos) == 0:
        return 0.0

    rangoX = maxX - minX if maxX > minX else 1.0
    rangoY = maxY - minY if maxY > minY else 1.0

    puntos_norm = []
    for p in puntos:
        nx = (p[0] - minX) / rangoX
        ny = (p[1] - minY) / rangoY
        puntos_norm.append((nx, ny))

    sortedPoints = sorted(puntos_norm, key=lambda p: p[0])

    hipervolumen = 0.0

    refX_norm = 1.0 
    refY_norm = 1.0

    for i in range(len(sortedPoints)):
        xx = sortedPoints[i][0]
        yy = sortedPoints[i][1]

        if xx <= refX_norm and yy <= refY_norm:
            if i + 1 < len(sortedPoints):
                nextX = sortedPoints[i + 1][0]
            else:
                nextX = refX_norm

            width = nextX - xx
            height = refY_norm - yy

            if width > 0 and height > 0:
                area = width * height
                hipervolumen += area

    return hipervolumen

def characterizeInstance(instanceContent):
    """Calculates statistics for the instance to match the report format."""
    tempAmpl = AMPL()
    tempAmpl.eval("reset;")
    tempAmpl.eval(modelo)
    tempAmpl.eval(instanceContent)

    fixedCosts = tempAmpl.getParameter("F").getValues().toDict()
    capacities = tempAmpl.getParameter("Cap").getValues().toDict()
    demand = tempAmpl.getParameter("d").getValues().toDict()
    transportCosts = tempAmpl.getParameter("TC").getValues().toDict()

    tempAmpl.close()

    fCosts = [f for f in fixedCosts.values()]
    caps = [c for c in capacities.values()]
    demands = [d for d in demand.values()]
    allTc = [tc for tc in transportCosts.values()]

    def getStats(name, values):
        meanVal = np.mean(values)
        stdVal = np.std(values)
        cvVal = (stdVal / meanVal) * 100 if meanVal != 0 else 0
        return (f"{name}:\n  - Promedio: {meanVal:.2f}\n"
                f"  - Desv. Estándar: {stdVal:.2f}\n"
                f"  - Coef. Variación (CV): {cvVal:.2f}%")

    totalCap = sum(caps)
    totalDem = sum(demands)
    ratio = totalCap / totalDem if totalDem > 0 else 0
    estado = "Muy Ajustada" if ratio < 1.5 else "Ajustada" if ratio < 3 else "Holgada"

    report = [
        "--- Análisis de la Instancia ---",
        getStats("Costos Fijos (F)", fCosts),
        getStats("Capacidades (Cap)", caps), 
        getStats("Demandas (d)", demands),
        getStats("Costos de Transporte (TC)", allTc), 
        "\nAnálisis de Tensión:", 
        f"  - Demanda Total de la Red: {totalDem:.2f}",
        f"  - Capacidad Total de CDs : {totalCap:.2f}", 
        f"  - Ratio (Cap/Dem)        : {ratio:.2f}x ({estado})",
        "--------------------------------"
    ]
    return "\n".join(report)

def exportData(instanceName, instance, amountOfCDs, epsilonData, gottenEpsilon, tplsData):
    """Generates a text report identical to the provided examples."""
    report = [
        "==================================================",
        "         REPORTE DE EJECUCIÓN MULTIOBJETIVO       ",
        "==================================================", 
        f"URL Instancia Evaluada: {instanceName}",
        f"Tamaño de Instancia: {amountOfCDs} CDs\n",
        "CARACTERIZACIÓN DE LA INSTANCIA",
        characterizeInstance(instance),
    ]

    if gottenEpsilon:
        report.extend([
            "\nRESULTADOS EPSILON-CONSTRAINT",
            f"Tiempo de ejecución : {epsilonData['time']:.4f} segundos",
            f"Hipervolumen        : {epsilonData['hv']:.4f}"
            f"\nPuntos Lexicográficos (Extremos del Frente):",
            f"  - Nadir: Transp={epsilonData['transMax']:.2f}, Infra={epsilonData['infraMax']:.2f}",
            f"  - Transp. Mín   : Transp={epsilonData['transMin']:.2f}, Infra={epsilonData['infraMax']:.2f}",
            f"  - Infra. Mín    : Transp={epsilonData['transMax']:.2f}, Infra={epsilonData['infraMin']:.2f}",
            f"\nPuntos del Frente ({len(epsilonData['paretoX'])} steps):"
        ])

        for i in range(len(epsilonData['paretoX'])):
            report.append(f"  Punto {i+1}: Transp={epsilonData['paretoX'][i]:.2f}, Infra={epsilonData['paretoY'][i]:.2f}")

    report.append("\nRESULTADOS HEURÍSTICA (TPLS)")
    if tplsData['stopped']:
        report.append(f"** El TPLS se detuvo prematuramente en la iteración {tplsData['stoppingIteration']} de {tplsData['amountIterations']} debido a falta de mejora. **")
    else:
        report.append(f"** El TPLS completó todas las iteraciones ({tplsData['amountIterations']}) sin detenerse por falta de mejora. **")
    report.append(f"Tiempo de ejecución : {tplsData['executionTime']:.4f} segundos")
    if gottenEpsilon:
        report.append(f"Hipervolumen        : {tplsData['hypervolume']:.4f}")
    report.append(f"\nFrente de Pareto Final - {len(tplsData['points'])} puntos:")
    
    for i, p in enumerate(tplsData['points']):
        report.append(f"  Punto {i+1}: Transp={p.Transport:.2f}, Infra={p.Infrastructure:.2f} | State: {p.state}")

    if gottenEpsilon:
        report.append("\nCOMPARATIVA ESTADÍSTICA")
        if epsilonData['hv'] > 0:
            calidad = (tplsData['hypervolume'] / epsilonData['hv']) * 100
            report.append(f"Calidad del TPLS vs Exacto : {calidad:.2f}% (Cobertura del Hipervolumen)")
            if tplsData['executionTime'] > 0:
                aceleracion = epsilonData['time'] / tplsData['executionTime'] 
                report.append(f"Aceleración de Tiempo      : El TPLS fue {aceleracion:.2f}x más rápido que Epsilon") 

    fileName = f"Reporte_{instanceName}_using_{tplsData['usedStrategy']}.txt" 
    with open(fileName, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"*** Reporte guardado en {fileName} ***") 

def readLexicographicData(filePath):
    data = {
        "infraLex": {"x": 0.0, "y": 0.0},
        "transpLex": {"x": 0.0, "y": 0.0},
        "paretoX": [],
        "paretoY": []
    }

    try:
        with open(filePath, 'r', encoding='utf-8') as file:
            content = file.read()

            # 1. Extrair Pontos Lexicográficos usando Regex
            # Busca o padrão "Punto X... [número]"
            infraX = re.search(r"Punto X lexicográfico de infraestructura e inventario:\s+([\d\.]+)", content)
            infraY = re.search(r"Punto Y lexicográfico de infraestructura e inventario:\s+([\d\.]+)", content)
            transpX = re.search(r"Punto X lexicográfico de transporte:\s+([\d\.]+)", content)
            transpY = re.search(r"Punto Y lexicográfico de transporte:\s+([\d\.]+)", content)

            if infraX: data["infraLex"]["x"] = float(infraX.group(1))
            if infraY: data["infraLex"]["y"] = float(infraY.group(1))
            if transpX: data["transpLex"]["x"] = float(transpX.group(1))
            if transpY: data["transpLex"]["y"] = float(transpY.group(1))

            # 2. Extrair listas Pareto X e Pareto Y
            # Busca o padrão "Pareto X: [ ... ]"
            listX = re.search(r"Pareto X:\s*(\[.*?\])", content)
            listY = re.search(r"Pareto Y:\s*(\[.*?\])", content)

            if listX:
                # ast.literal_eval converte a string da lista em uma lista real de Python com segurança
                data["paretoX"] = ast.literal_eval(listX.group(1))
            if listY:
                data["paretoY"] = ast.literal_eval(listY.group(1))

        return data

    except FileNotFoundError:
        print(f"Error: El archivo {filePath} no fue encontrado.")
        return None
    except Exception as e:
        print(f"Error al procesar el archivo: {e}")
        return None

def loadDatInstance(name):

    if not name.endswith(".dat"):
        fileName = f"{name}.dat"
    else:
        fileName = name
        
    folderName = "instances"
    filePath = os.path.join(folderName, fileName)
    
    try:
        with open(filePath, 'r', encoding='utf-8') as fileObject:
            fileContent = fileObject.read()
        return fileContent
    except FileNotFoundError:
        print(f"Error: El archivo '{fileName}' no se encontró en la carpeta '{folderName}'.")
        return None
    except Exception as errorObject:
        print(f"Error inesperado al leer la instancia: {errorObject}")
        return None

def loadConfig(configPath):
    with open(configPath, 'r', encoding='utf-8') as file:
        return json.load(file)