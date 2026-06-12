from matplotlib.dates import TH
import numpy as np
import requests
import matplotlib.pyplot as plt
import sys
import os
from time import time
from collections import deque

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.instanceGenerator import generateInstance, saveVisualData
from src.plotting import plotInstanceMap, plotUrbanHeatmap

if __name__ == "__main__":
    generatorConfig = loadConfig("generator_config.json")

    cds, clients, centroids = generateInstance(
        numClients=100, 
        planeSize=100, 
        unitTransportCost=2.12, 
        seed=42, 
        clustered=True, 
        numClusters=4, 
        stdDev=8, 
        baseMinCost=15000, 
        maxPremium=25000, 
        baseMinCapacity=1100, 
        maxCapacity=2500
    )  

    saveVisualData(cds, clients, centroids, "instance_data.json")
    plotInstanceMap(cds, clients, centroids, areaSize=100, fileName="instance_map.png")
    plotUrbanHeatmap(centroids, areaSize=100, fileName="urban_heatmap.png")