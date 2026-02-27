from amplpy import AMPL, ampl_notebook
from src.model import cd, client, paretoPoint, modelo

ampl = AMPL()
ampl.eval(modelo)
ampl.setOption("solver","gurobi")
ampl.option['gurobi_options'] = 'NonConvex=2 MIPGap=0.05'

def instanceToAmpl(cdList, clientList, k, th):
    I_set = [c.id for c in cdList]
    J_set = [cl.id for cl in clientList]

    lines = ["data;"]
    lines.append("set I := " + " ".join(map(str, I_set)) + ";")
    lines.append("set J := " + " ".join(map(str, J_set)) + ";")

    lines.append("param F := " + " ".join(f"{c.id} {c.fixedCost}" for c in cdList) + ";")
    lines.append("param Cap := " + " ".join(f"{c.id} {c.capacity}" for c in cdList) + ";")
    lines.append("param RC := " + " ".join(f"{c.id} {c.replenishmentCost}" for c in cdList) + ";")
    lines.append("param OC := " + " ".join(f"{c.id} {c.reorderCost}" for c in cdList) + ";")
    lines.append("param HC := " + " ".join(f"{c.id} {c.holdingCost}" for c in cdList) + ";")
    lines.append("param LT := " + " ".join(f"{c.id} {c.leadTime}" for c in cdList) + ";")

    lines.append("param d := " + " ".join(f"{cl.id} {cl.demand}" for cl in clientList) + ";")
    lines.append("param u := " + " ".join(f"{cl.id} {cl.variance}" for cl in clientList) + ";")

    lines.append(f"param K := {k};")
    lines.append(f"param TH := {th};")
    lines.append("param TC := ")

    for cl in clientList:
        for i, cost in enumerate(cl.transportCost):
            lines.append(f"{i} {cl.id} {cost}")

    lines.append(";")

    return "\n".join(lines)

def calculateFitness(cdList, clientList, K, TH, statesList):
    paretoPoints = []

    for state in statesList:
        amplDataFix = instanceToAmpl(cdList, clientList, K, TH)

        ampl.eval("reset data;")
        ampl.eval(amplDataFix)
        ampl.eval("unfix Z;")

        for i, val in enumerate(state):
            ampl.eval(f"fix Z[{i}] := {val};")

        ampl.solve()
        
        infra_cost = ampl.get_variable("InfrastructureCost").value()

        trans_cost = ampl.get_variable("TransportCost").value()

        paretoPoints.append(paretoPoint(infra_cost, trans_cost, tuple(state)))

    return paretoPoints