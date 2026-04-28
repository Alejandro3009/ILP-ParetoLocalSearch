from amplpy import OutputHandler

class paretoPoint:
    def __init__(self, Infrastructure, Transport, state, explored):
        self.Infrastructure = Infrastructure
        self.Transport = Transport
        self.state = state
        self.explored = explored #Solo se utiliza en el PLS con busqueda en un unico punto, para marcar los puntos que ya han sido explorados

class cd:
  def __init__(self, id, capacity, fixedCost, reorderCost, holdingCost, leadTime, replenishmentCost):
    self.id = id
    self.capacity = capacity
    self.fixedCost = fixedCost
    self.reorderCost = reorderCost
    self.holdingCost = holdingCost
    self.leadTime = leadTime
    self.replenishmentCost = replenishmentCost

    self.assignedDemand = 0
    self.assignedVariance = 0
    self.open = False

class client:
  def __init__(self, id, demand, variance):
    self.id = id
    self.demand = demand
    self.variance = variance
    self.transportCost = []
    self.assignedCd = None

    def getCost(cdId):
      return self.transportCost[cdId]
  
class movements: #clase utilizada unicamente en el tabu pareto local
    def __init__(self, changeState, moves):
        self.changeState = changeState
        self.moves = moves

class SilentOutputHandler(OutputHandler):
    def output(self, kind, msg):
        pass

modelo = r"""
set I;   # Centros de distribución
set J;   # Clientes

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
param Alpha default 0.5; # Parámetro para ponderar los objetivos

param maxInfra default 1;
param maxTransp default 1;

var Z{i in I} >= 0, <= 1, integer;
var Y{i in I,j in J} >= 0, <= 1, integer;
var D{i in I} >= 0;
var U{i in I} >= 0;

var QD{i in I} >= 0;
var QU{i in I} >= 0;

var InfrastructureCost = 
    sum{i in I} F[i] * Z[i] +
    sum{i in I} TH * sqrt(2 * HC[i] * OC[i]) * QD[i] +
    sum{i in I} TH * HC[i] * K * sqrt(LT[i]) * QU[i];

var TransportCost = sum{i in I, j in J} TH * (RC[i] + TC[i,j]) * d[j] * Y[i,j];

minimize TotalCost:
    (InfrastructureCost/maxInfra) * Alpha + (TransportCost/maxTransp) * (1-Alpha);

s.t. Assign{j in J}: sum{i in I} Y[i,j] = 1;
s.t. Capacity{i in I}: sum{j in J} d[j]*Y[i,j] <= Cap[i]*Z[i];
s.t. DemandDef{i in I}: D[i] = sum{j in J} d[j]*Y[i,j];
s.t. VarDef{i in I}: U[i] = sum{j in J} u[j]*Y[i,j];

# Restricciones Cuadráticas
s.t. QuadDemand{i in I}: QD[i] * QD[i] >= D[i];
s.t. QuadVar{i in I}: QU[i] * QU[i] >= U[i];

"""