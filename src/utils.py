import re
import json
import random
import numpy as np
import time
from collections import deque
from src.model import cd, client

def loadJsonInstance(instance_name, filepath="instancias pequeños valores.json"):
    with open(filepath, "r") as f:
        data = json.load(f)

    if instance_name not in data:
        raise ValueError(f"Instancia '{instance_name}' no encontrada. Opciones: {list(data.keys())}")

    raw_cds = data[instance_name]["cds"]
    raw_clients = data[instance_name]["clients"]

    cdList = []
    clientList = []

    # Reconstruir objetos CD
    for d in raw_cds:
        new_cd = cd(d["id"], d["capacity"], d["fixedCost"], d["reorderCost"],
                    d["holdingCost"], d["leadTime"], d["replenishmentCost"])
        cdList.append(new_cd)

    # Reconstruir objetos Cliente
    for c in raw_clients:
        new_cl = client(c["id"], c["demand"], c["variance"])
        new_cl.transportCost = c["transportCost"] # Asignar la lista de costos
        clientList.append(new_cl)

    print(f"Instancia '{instance_name}' cargada: {len(cdList)} CDs, {len(clientList)} Clientes.")
    return cdList, clientList

def loadTextInstance(currentInstance):
    # Remove comments to avoid parsing issues
    clean_text = re.sub(r'#.*', '', currentInstance)

    def get_scalar(name):
        match = re.search(rf'param\s+{name}\s*:=\s*([\d\.]+)\s*;', clean_text, re.IGNORECASE)
        return float(match.group(1)) if match else 0.0

    def get_map(name):
        pattern = rf'param\s+{name}(?:\s*\[.*?\])?\s*:=\s*(.*?);'
        match = re.search(pattern, clean_text, re.DOTALL | re.IGNORECASE)
        if not match: return {}
        # Captures pairs of index and value
        return {int(m[0]): float(m[1]) for m in re.findall(r'(\d+)\s+([\d\.\-]+)', match.group(1))}

    # 1. Extract Dimensions and Scalars
    K = get_scalar('K')
    TH = get_scalar('TH')
    f_map = get_map('f')
    cap_map = get_map('Cap')
    rc_map = get_map('RC')
    oc_map = get_map('OC')
    hc_map = get_map('HC')
    lt_map = get_map('LT')
    d_map = get_map('d')
    u_map = get_map('u')

    # Calculate counts based on the highest index found in the data
    num_cds = max(f_map.keys()) + 1 if f_map else 0
    num_clients = max(d_map.keys()) + 1 if d_map else 0

    # 2. Extract TC Triplets (CD_ID, Client_ID, Cost)
    # We find the block and look for groups of 3 numbers
    tc_lookup = {} # Format: {(cd_id, client_id): cost}
    tc_pattern = r'param\s+TC(?:\s*\[.*?\])?\s*:=\s*(.*?);'
    tc_match = re.search(tc_pattern, clean_text, re.DOTALL | re.IGNORECASE)

    if tc_match:
        # Find all numbers and group them into triplets
        all_nums = re.findall(r'[\d\.\-]+', tc_match.group(1))
        for i in range(0, len(all_nums), 3):
            if i + 2 < len(all_nums):
                cd_idx = int(float(all_nums[i]))
                cl_idx = int(float(all_nums[i+1]))
                cost = float(all_nums[i+2])
                tc_lookup[(cd_idx, cl_idx)] = cost

    # 3. Instantiate CD Objects
    cds_list = [
        cd(id=i, capacity=cap_map.get(i, 0.0), fixedCost=f_map.get(i, 0.0),
           reorderCost=oc_map.get(i, 0.0), holdingCost=hc_map.get(i, 0.0),
           leadTime=lt_map.get(i, 0.0), replenishmentCost=rc_map.get(i, 0.0))
        for i in range(num_cds)
    ]

    # 4. Instantiate Client Objects
    clients_list = []
    for j in range(num_clients):
        cl = client(id=j, demand=d_map.get(j, 0.0), variance=u_map.get(j, 0.0))

        # Populate the array where index i is the cost from CD i
        # This matches your logic: TC[10, 2, 5] means cost from CD 0 is 10, CD 1 is 2...
        tc_array = []
        for i in range(num_cds):
            tc_array.append(tc_lookup.get((i, j), 0.0))

        cl.transportCost = tc_array
        clients_list.append(cl)

    return cds_list, clients_list, K, TH

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

def randomSolution(cdList, clientList):
    for client in clientList:
        chosen = random.choice(cdList)
        client.assignedCd = chosen.id
        chosen.open = True
        chosen.assignedDemand += client.demand
        chosen.assignedVariance += client.variance

def getStateTuple(cdList):
    aux = list(c.open for c in cdList)
    for i in range(len(aux)):
        if aux[i] == True:
            aux[i] = 1
        else:
            aux[i] = 0
    return tuple(aux)
    
def getTotalDemand(clientList):
    totalDemand = 0
    totalVariance = 0
    for client in clientList:
        totalDemand += client.demand
        totalVariance += client.variance
    return totalDemand + totalVariance

def calcularHipervolumen(puntos, refX, refY):
    if len(puntos) == 0:
        return 0.0

    sortedPoints = sorted(puntos, key=lambda p: p[0])
    print(sortedPoints)
    hipervolumen = 0.0

    for i in range(len(sortedPoints)):
        xx = sortedPoints[i][0]
        yy = sortedPoints[i][1]

        if xx <= refX or yy <= refY:
            if i + 1 < len(sortedPoints):
                nextX = sortedPoints[i + 1][0]
            else:
                nextX = refX

            width = nextX - xx
            height = refY - yy

            area = width * height
            hipervolumen += area

    return hipervolumen

def characterizeInstance(cdList, clientList):
    """Calculates statistics for the instance to match the report format."""
    fCosts = [c.fixedCost for c in cdList]
    caps = [c.capacity for c in cdList]
    demands = [cl.demand for cl in clientList]
    allTc = [cost for cl in clientList for cost in cl.transportCost]

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

def exportData(instanceUrl, cdList, clientList, epsilonData, gottenEpsilon, tplsData, instanceName):
    """Generates a text report identical to the provided examples."""
    report = [
        "==================================================",
        "         REPORTE DE EJECUCIÓN MULTIOBJETIVO       ",
        "==================================================", 
        f"URL Instancia Evaluada: {instanceUrl}",
        f"Tamaño de Instancia: {len(cdList)} CDs\n",
        "CARACTERIZACIÓN DE LA INSTANCIA",
        characterizeInstance(cdList, clientList),
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

    fileName = f"Reporte_{instanceName}_{int(time.time())}.txt" 
    with open(fileName, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"*** Reporte guardado en {fileName} ***") 