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

s.t. Assign{j in J}: sum{i in I} Y[i,j] = 1;
s.t. Capacity{i in I}: sum{j in J} d[j]*Y[i,j] <= Cap[i]*Z[i];
s.t. DemandDef{i in I}: D[i] = sum{j in J} d[j]*Y[i,j];
s.t. VarDef{i in I}: U[i] = sum{j in J} u[j]*Y[i,j];

s.t. QuadDemand{i in I}: QD[i] * QD[i] >= D[i];
s.t. QuadVar{i in I}: QU[i] * QU[i] >= U[i];