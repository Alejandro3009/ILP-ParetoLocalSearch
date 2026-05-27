import concurrent.futures
import random
from time import time
from amplpy import AMPL
from src.model import cd, client, paretoPoint, SilentOutputHandler, modelo

ampl = AMPL()
ampl.eval(modelo)
ampl.setOption("solver","gurobi")
ampl.option['gurobi_options'] = 'NonConvex=2 MIPGap=0.05'
ampl.setOutputHandler(SilentOutputHandler())

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
        for i, cost in cl.transportCost.items():
            lines.append(f"{i} {cl.id} {cost}")

    lines.append(";")

    return "\n".join(lines)

def rebalanceStates(state, fixCosts, asignacion, infra_cost):
    for i in range(len(asignacion)):
        if asignacion[i][1] == 0 and state[i] == 1:
            state[i] = 0
            infra_cost -= fixCosts[i]
    return state, infra_cost

def solve_single_state(args):
    """Worker function: Solves one state in a private AMPL instance."""
    instance, state, alphaValue, lexPoints = args

    # Each process MUST have its own AMPL object
    worker_ampl = AMPL()
    worker_ampl.eval(modelo)
    worker_ampl.setOption("solver", "gurobi") 
    worker_ampl.setOption("presolve", 0)
    worker_ampl.setOption("gurobi_options", "NonConvex=2 MIPGap=0.05")
    worker_ampl.setOutputHandler(SilentOutputHandler())

    # Set data and fix Z variables [cite: 71]
    worker_ampl.eval(instance)
    for i, val in enumerate(state):
        worker_ampl.eval(f"fix Z[{i}] := {val};")

    worker_ampl.param['Alpha'] = alphaValue

    if lexPoints is not None:
        worker_ampl.param['maxInfra'] = lexPoints[1]
        worker_ampl.param['maxTransp'] = lexPoints[0]

    worker_ampl.solve()
    
    if worker_ampl.getValue("solve_result") != "solved":
        print(f"Warning: State {state} could not be solved. Result: {worker_ampl.getValue('solve_result')}")
        return None, worker_ampl.getValue("solve_result")

    cdsFixedCost = worker_ampl.getParameter("F").getValues().toDict()
    infra_cost = worker_ampl.get_variable("InfrastructureCost").value() 
    trans_cost = worker_ampl.get_variable("TransportCost").value()
    asignacion = worker_ampl.get_variable("D").get_values().toList()
    solveResult = worker_ampl.getValue("solve_result")
    
    # Close session to free memory
    worker_ampl.close()
    
    # Use your existing rebalance logic
    new_state, new_infra = rebalanceStates(list(state), cdsFixedCost, asignacion, infra_cost)

    return paretoPoint(new_infra, trans_cost, tuple(new_state), False), solveResult

def calculateFitnessParallel(instance, statesList, max_workers=10, alphaValue=0.5, lexPoints=None):
    """Parallel coordinator."""
    time0 = time()
    
    # Prepare arguments for each worker
    tasks = [(instance, state, alphaValue, lexPoints) for state in statesList]
    
    paretoPoints = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Map tasks to workers
        results = list(executor.map(solve_single_state, tasks))

        for result in results:
            if result[0] is None:
                print("Warning: One of the states could not be solved.")
            else:
                paretoPoints.append(result[0])

        print(f"amount of states solved: {len(paretoPoints)} out of {len(results)}")
    
    time1 = time()
    return paretoPoints, time1 - time0

def fixAssignment(state):
    return [1 if val[1] >= 0.3 else 0 for val in state]

def solve_single_relax_state(args):
    instanceContent, state, fixingSize, alphaValue, lexPoints, tabuList = args
    
    # Each process MUST have its own AMPL object
    worker_ampl = AMPL()
    worker_ampl.eval(modelo)
    worker_ampl.setOption("solver", "gurobi") 
    worker_ampl.setOption("presolve", 0)
    worker_ampl.setOption("gurobi_options", "NonConvex=2 MIPGap=0.05")
    worker_ampl.setOption("relax_integrality", 1)
    worker_ampl.setOutputHandler(SilentOutputHandler())

    # Set data and fix Z variables [cite: 71]
    worker_ampl.eval(instanceContent)

    # Identify which CDs are NOT tabu
    # We only want to fix variables that are NOT currently restricted by Tabu
    nonTabuIndices = [i for i in range(len(state)) if tabuList.get(i) is None]
    
    # Calculate how many to fix based on the available non-tabu population
    kToFix = min(len(nonTabuIndices), int(len(state) * fixingSize))
    indicesToFix = random.sample(nonTabuIndices, k=kToFix)

    for i in indicesToFix:
        worker_ampl.eval(f"fix Z[{i}] := {state[i]};")

    worker_ampl.param['Alpha'] = alphaValue

    if lexPoints is not None:
        worker_ampl.param['maxInfra'] = lexPoints[1]
        worker_ampl.param['maxTransp'] = lexPoints[0]

    worker_ampl.solve()
    
    if worker_ampl.getValue("solve_result") != "solved":
        print(f"Warning: State {state} could not be solved. Result: {worker_ampl.getValue('solve_result')}")
        return None, worker_ampl.getValue("solve_result")

    asignacionZ = worker_ampl.get_variable("Z").get_values().toList()
    solveResult = worker_ampl.getValue("solve_result")
    
    print(asignacionZ)

    # Close session to free memory
    worker_ampl.close()
    
    # Use your existing rebalance logic
    fixState = fixAssignment(asignacionZ)

    return  tuple(fixState), solveResult

def parallelLinearRelaxation(instanceContent, statesList, fixingSize, max_workers=10, alphaValue=0.5, lexPoints=None, tabuList=None):
    """Parallel coordinator."""
    time0 = time()
    
    # Prepare arguments for each worker
    tasks = [(instanceContent, state, fixingSize, alphaValue, lexPoints, tabuList) for state in statesList]
    
    relaxStates = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Map tasks to workers
        results = list(executor.map(solve_single_relax_state, tasks))

        for result in results:
            if result[1] is None:
                print("Warning: One of the states could not be solved.")
            else:
                relaxStates.append(result[0])
    
    time1 = time()
    return relaxStates, time1 - time0



#def calculateFitness(cdList, clientList, K, TH, statesList, alphaValue):
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

        state, infra_cost = rebalanceStates(list(state), cdList, asignacion, infra_cost)

        print(state, infra_cost)

        paretoPoints.append(paretoPoint(infra_cost, trans_cost, tuple(state), False))

    time1 = time()
    return paretoPoints, time1 - time0
