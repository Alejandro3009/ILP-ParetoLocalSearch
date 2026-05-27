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
from src.initialSolution import generateInitialSolution
from src.utils import printSummary, calcularHipervolumen, exportData, loadDatInstance
from src.solver import calculateFitnessParallel, instanceToAmpl
from src.TPLS_MPS import MultiPointParetoSearch
from src.TPLS_OPS import onePointParetoSearch
from src.model import modelo
from lexsrc.model import mTransport, mInfrastructure
from lexsrc.solver import solveInstance, solveEpsilon, filterEpsilonFront
from lexsrc.utils import saveEpsilonFront, loadEpsilonResults
from src.instanceGenerator import generateInstance, saveVisualData
from src.plotting import plotInstanceMap, plotUrbanHeatmap

if __name__ == "__main__":

    currentInstance = loadDatInstance("20x10-0.dat")

    tempAmpl = AMPL()
    tempAmpl.eval("reset;")
    tempAmpl.eval(modelo)
    tempAmpl.eval(currentInstance)
    
    cds = tempAmpl.getSet("I").getValues().toList()
    clients = tempAmpl.getSet("J").getValues().toList()

    print(f"El número de centros de distribución es {cds}")
    print(f"El número de clientes es {len(clients)}")

    clientsTransportCost = tempAmpl.getParameter("TC").getValues().toDict()

    print(f"El coste de transporte del centro 0 al cliente 0 es {clientsTransportCost[0,0]}")

    for tc in clientsTransportCost.values():
        print(tc)

    cdsFixedCost = tempAmpl.getParameter("F").getValues()
    cdsFixedCost = cdsFixedCost.toDict()

    print(f"El coste fijo del centro 0 es {len(cdsFixedCost)}")

    print (len(clientsTransportCost))