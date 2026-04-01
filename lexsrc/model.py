import numpy as np
import random

mInfrastructure = r"""
set I; 
set J; 

param F{i in I}; 
param Cap{i in I};
param d{j in J};
param u{j in J};
param RC{i in I};
param TC{i in I,j in J};
param OC{i in I};
param HC{i in I};
param LT{i in I};
param K;
param TH;
param epsilon;

var Z{i in I} binary;
var Y{i in I,j in J} binary;
var D{i in I} >= 0;
var U{i in I} >= 0;

var QD{i in I} >= 0;
var QU{i in I} >= 0;

var CostoInfra = 
    sum{i in I} F[i] * Z[i] +                        
    sum{i in I} TH * sqrt(2 * HC[i] * OC[i]) * QD[i] + 
    sum{i in I} TH * HC[i] * K * sqrt(LT[i]) * QU[i];  

var CostoTransp = sum{i in I, j in J} TH * (RC[i] + TC[i,j]) * d[j] * Y[i,j];

minimize TotalCost:
    CostoInfra;

s.t. epsilonConstraint: CostoTransp <= epsilon;
s.t. Assign{j in J}: sum{i in I} Y[i,j] = 1;
s.t. Capacity{i in I}: sum{j in J} d[j]*Y[i,j] <= Cap[i]*Z[i];
s.t. DemandDef{i in I}: D[i] = sum{j in J} d[j]*Y[i,j];
s.t. VarDef{i in I}: U[i] = sum{j in J} u[j]*Y[i,j];

s.t. QuadDemand{i in I}: QD[i] * QD[i] >= D[i];
s.t. QuadVar{i in I}: QU[i] * QU[i] >= U[i];

"""

mTransport = r"""
set I; 
set J; 

param F{i in I}; 
param Cap{i in I};
param d{j in J};
param u{j in J};
param RC{i in I};
param TC{i in I,j in J};
param OC{i in I};
param HC{i in I};
param LT{i in I};
param K;
param TH;
param epsilon;

var Z{i in I} binary;
var Y{i in I,j in J} binary;
var D{i in I} >= 0;
var U{i in I} >= 0;

var QD{i in I} >= 0;
var QU{i in I} >= 0;

var CostoInfra = 
    sum{i in I} F[i] * Z[i] +                        
    sum{i in I} TH * sqrt(2 * HC[i] * OC[i]) * QD[i] + 
    sum{i in I} TH * HC[i] * K * sqrt(LT[i]) * QU[i];  

var CostoTransp = sum{i in I, j in J} TH * (RC[i] + TC[i,j]) * d[j] * Y[i,j];

minimize TotalCost:
    CostoTransp;

s.t. epsilonConstraint: CostoInfra <= epsilon;
s.t. Assign{j in J}: sum{i in I} Y[i,j] = 1;
s.t. Capacity{i in I}: sum{j in J} d[j]*Y[i,j] <= Cap[i]*Z[i];
s.t. DemandDef{i in I}: D[i] = sum{j in J} d[j]*Y[i,j];
s.t. VarDef{i in I}: U[i] = sum{j in J} u[j]*Y[i,j];

s.t. QuadDemand{i in I}: QD[i] * QD[i] >= D[i];
s.t. QuadVar{i in I}: QU[i] * QU[i] >= U[i];
"""

#---------------------------------Clases y funciones para instancias aleatorias----------------------------------
class Cd:
    def __init__(self, id, capacity, fixedCost, reorderCost, holdingCost, leadTime, replenishmentCost):
        self.id = id
        self.capacity = capacity
        self.fixedCost = fixedCost
        self.reorderCost = reorderCost
        self.holdingCost = holdingCost
        self.leadTime = leadTime
        self.replenishmentCost = replenishmentCost

class Client:
    def __init__(self, id, demand, variance):
        self.id = id
        self.demand = demand
        self.variance = variance
        self.transportCost = []

def randomInstance(size, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    cdList = []
    clientList = []

    for i in range(size):
        capacity = np.random.randint(100, 300)
        fixedCost = np.random.randint(500, 1000)
        reorderCost = np.random.randint(50, 150)
        holdingCost = np.random.uniform(1, 20)
        leadTime = np.random.randint(1, 4)
        replenishmentCost = np.random.randint(50, 150)
        cdList.append(Cd(i, capacity, fixedCost, reorderCost, holdingCost, leadTime, replenishmentCost))

    for j in range(size):
        demand = np.random.randint(10, 50)
        variance = np.random.randint(5, 20)
        newClient = Client(j, demand, variance)
        for warehouse in cdList:
            newClient.transportCost.append(np.random.randint(50, 250))
        clientList.append(newClient)

    return cdList, clientList

def instanceToAmpl(cdList, clientList, kVal, thVal):
    iSet = [c.id for c in cdList]
    jSet = [cl.id for cl in clientList]

    lines = ["data;"]
    lines.append("set I := " + " ".join(map(str, iSet)) + ";")
    lines.append("set J := " + " ".join(map(str, jSet)) + ";")

    lines.append("param F := " + " ".join(f"{c.id} {c.fixedCost}" for c in cdList) + ";")
    lines.append("param Cap := " + " ".join(f"{c.id} {c.capacity}" for c in cdList) + ";")
    lines.append("param RC := " + " ".join(f"{c.id} {c.replenishmentCost}" for c in cdList) + ";")
    lines.append("param OC := " + " ".join(f"{c.id} {c.reorderCost}" for c in cdList) + ";")
    lines.append("param HC := " + " ".join(f"{c.id} {c.holdingCost}" for c in cdList) + ";")
    lines.append("param LT := " + " ".join(f"{c.id} {c.leadTime}" for c in cdList) + ";")

    lines.append("param d := " + " ".join(f"{cl.id} {cl.demand}" for cl in clientList) + ";")
    lines.append("param u := " + " ".join(f"{cl.id} {cl.variance}" for cl in clientList) + ";")

    lines.append(f"param K := {kVal};")
    lines.append(f"param TH := {thVal};")
    lines.append("param TC := ")

    for cl in clientList:
        for i, cost in enumerate(cl.transportCost):
            lines.append(f"{i} {cl.id} {cost}")

    lines.append(";")
    return "\n".join(lines)