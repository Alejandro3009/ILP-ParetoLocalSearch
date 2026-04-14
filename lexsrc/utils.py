import json
import os
import sys

def saveEpsilonFront(filePath, instanceUrl, epsilonData):
    allData = {}

    if os.path.exists(filePath):
        with open(filePath, 'r') as file:
            allData = json.load(file)

    instanceEntry = {
        "metadata": {
            "executionTime": epsilonData['time'],
            "hypervolume": epsilonData['hv'],
        },
        "lexicographicPoints": {
            "infraMin": {"transp": epsilonData['transMax'], "infra": epsilonData['infraMin']},
            "infraMax": {"transp": epsilonData['transMin'], "infra": epsilonData['infraMax']}
        },
        "paretoFront": {
            "x": epsilonData['paretoX'], # Transport
            "y": epsilonData['paretoY']  # Infrastructure
        }
    }

    allData[instanceUrl] = instanceEntry

    with open(filePath, 'w', encoding='utf-8') as f:
        json.dump(allData, f, indent=4)

    print(f" --- Datos de Epsilon actualizados exitosamente en '{filePath} --- '")

def loadEpsilonResults(filePath, instanceUrl):
    if not os.path.exists(filePath):
        return None

    try:
        with open(filePath, 'r', encoding='utf-8') as f:
            allResults = json.load(f)
            
        if instanceUrl in allResults:
            print(f" --- Resultados previos encontrados para: {instanceUrl} --- ")
            data = allResults[instanceUrl]
            
            return {
                'time': data['metadata']['executionTime'],
                'hv': data['metadata']['hypervolume'],
                'transMin': data['lexicographicPoints']['infraMax']['transp'],
                'transMax': data['lexicographicPoints']['infraMin']['transp'],
                'infraMin': data['lexicographicPoints']['infraMin']['infra'],
                'infraMax': data['lexicographicPoints']['infraMax']['infra'],
                'paretoX': data['paretoFront']['x'],
                'paretoY': data['paretoFront']['y']
            }
    except Exception as e:
        print(f"Error al cargar resultados previos: {e}")
        
    return None