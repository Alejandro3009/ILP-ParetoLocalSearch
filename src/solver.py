import concurrent.futures
from time import time
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

def rebalanceStates(state, cdList, asignacion, infra_cost):
    k = [cd.fixedCost for cd in cdList]
    for i in range(len(asignacion)):
        if asignacion[i][1] == 0 and state[i] == 1:
            state[i] = 0
            infra_cost -= k[i]
    return state, infra_cost

def solve_single_state(args):
    """Worker function: Solves one state in a private AMPL instance."""
    state, cdList, clientList, K, TH, alphaValue = args
    
    # Each process MUST have its own AMPL object
    worker_ampl = AMPL()
    worker_ampl.eval(modelo)
    worker_ampl.setOption("solver", "gurobi") 
    worker_ampl.setOption("gurobi_options", "NonConvex=2 MIPGap=0.05")
    
    # Set data and fix Z variables [cite: 71]
    amplDataFix = instanceToAmpl(cdList, clientList, K, TH)
    worker_ampl.eval(amplDataFix)
    for i, val in enumerate(state):
        worker_ampl.eval(f"fix Z[{i}] := {val};")

    worker_ampl.param['Alpha'] = alphaValue

    worker_ampl.solve()
    
    infra_cost = worker_ampl.get_variable("InfrastructureCost").value() 
    trans_cost = worker_ampl.get_variable("TransportCost").value()
    asignacion = worker_ampl.get_variable("D").get_values().toList()
    
    # Close session to free memory
    worker_ampl.close()
    
    # Use your existing rebalance logic
    new_state, new_infra = rebalanceStates(list(state), cdList, asignacion, infra_cost)
    return paretoPoint(new_infra, trans_cost, tuple(new_state))

def calculateFitness(cdList, clientList, K, TH, statesList, alphaValue):
    paretoPoints = []

    time0 = time() 
    for state in statesList:
        amplDataFix = instanceToAmpl(cdList, clientList, K, TH)

        ampl.eval("reset data;")
        ampl.eval(amplDataFix)
        ampl.eval("unfix Z;")

        for i, val in enumerate(state):
            ampl.eval(f"fix Z[{i}] := {val};")

        ampl.param['Alpha'] = alphaValue

        ampl.solve()
        
        infra_cost = ampl.get_variable("InfrastructureCost").value()

        trans_cost = ampl.get_variable("TransportCost").value()

        asignacion = ampl.get_variable("D").get_values().toList()

        print("Asignacion:", asignacion)

        state, infra_cost = rebalanceStates(list(state), cdList, asignacion, infra_cost)

        print(state, infra_cost)

        paretoPoints.append(paretoPoint(infra_cost, trans_cost, tuple(state)))

    time1 = time()
    return paretoPoints, time1 - time0

def calculateFitnessParallel(cdList, clientList, K, TH, statesList, max_workers=10, alphaValue=0.5):
    """Parallel coordinator."""
    time0 = time()
    
    # Prepare arguments for each worker
    tasks = [(state, cdList, clientList, K, TH, alphaValue) for state in statesList]
    
    paretoPoints = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Map tasks to workers
        paretoPoints = list(executor.map(solve_single_state, tasks))
    
    time1 = time()
    return paretoPoints, time1 - time0