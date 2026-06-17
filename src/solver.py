import concurrent.futures
import random
import re
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
    instance, state, alphaValue, lexPoints, solver = args

    # Each process MUST have its own AMPL object
    worker_ampl = AMPL()
    worker_ampl.eval(modelo)

    match solver:
        case "gurobi":
            worker_ampl.setOption("solver", "gurobi") 
            worker_ampl.setOption("presolve", 0)
            worker_ampl.setOption("gurobi_options", "NonConvex=2 MIPGap=0.05 outlev=0")
            worker_ampl.setOption('output', 0)
        case "knitro":
            options = "outlev=0 mip_integral_gap_rel=0.05 opttol=1e-4 feastol=1e-4 mip_method=1 outlev=0"
            worker_ampl.setOption("knitro_options", options)
            worker_ampl.setOption('output', 0)
        case _:
            raise ValueError("El solver seleccionado no es reconocido.")

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
    
    # LINEAS DE DEPURACIÓN TEMPORAL:
    status = worker_ampl.getValue("solve_result")
    if status != "solved":
        _ = None
        #print(f"\n[AMPL ERROR] Estado: {state} | Status: {status} | Msg: {worker_ampl.getValue('solve_message')}")

    cdsFixedCost = worker_ampl.getParameter("F").getValues().toDict()
    infra_cost = worker_ampl.get_variable("InfrastructureCost").value() 
    trans_cost = worker_ampl.get_variable("TransportCost").value()
    asignacion = worker_ampl.get_variable("D").get_values().toList()
    solveResult = worker_ampl.getValue("solve_result")

    solveMsg = worker_ampl.getValue("solve_message")

    iterations, branchNodes = getInfo(solveMsg, solver)
    
    # Close session to free memory
    worker_ampl.close()
    
    # Use your existing rebalance logic
    new_state, new_infra = rebalanceStates(list(state), cdsFixedCost, asignacion, infra_cost)

    return paretoPoint(new_infra, trans_cost, tuple(new_state), False), solveResult, iterations, branchNodes

def calculateFitnessParallel(instance, statesList, solver, max_workers=10, alphaValue=0.5, lexPoints=None):
    """Parallel coordinator."""
    time0 = time()

    clean_states = []
    for state in statesList:
        clean_states.append(tuple(int(val) for val in state))
    
    # Prepare arguments for each worker
    tasks = [(instance, state, alphaValue, lexPoints, solver) for state in statesList]
    
    paretoPoints = []
    iterations = 0
    branchNodes = 0
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        try:
            # Dynamically execute worker tasks in isolated worker subprocesses
            results = list(executor.map(solve_single_state, tasks))
            
            # FIXED: Properly unpack and check the processed results array
            for result in results:
                if result is None or result[0] is None:
                    continue
                if result[1] != "solved":
                    continue
                
                # Append individual verified points into your tracking front array
                paretoPoints.append(result[0])
                iterations += result[2]
                branchNodes += result[3]
                
        except Exception as e:
            import traceback
            print(f"\nCRITICAL THREAD ERROR EXPOSED: {str(e)}")
            traceback.print_exc()
            raise e

    time1 = time()
    return paretoPoints, time1 - time0, iterations, branchNodes

def fixAssignment(state):
    return [1 if val[1] >= 0.3 else 0 for val in state]

def solve_single_relax_state(args):
    instanceContent, state, fixingSize, alphaValue, lexPoints, tabuList, solver = args
    
    # Each process MUST have its own AMPL object
    worker_ampl = AMPL()
    worker_ampl.eval(modelo)

    match solver:
        case "gurobi":
            worker_ampl.setOption("solver", "gurobi") 
            worker_ampl.setOption("presolve", 0)
            worker_ampl.setOption("gurobi_options", "NonConvex=2 MIPGap=0.05 outlev=0")
            worker_ampl.setOption('output', 0)
        case "knitro":
            options = "outlev=0 mip_integral_gap_rel=0.05 opttol=1e-4 feastol=1e-4 mip_method=1 outlev=0"
            worker_ampl.setOption("knitro_options", options)
            worker_ampl.setOption('output', 0)
        case _:
            raise ValueError("El solver seleccionado no es reconocido.")

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
    
    # LINEAS DE DEPURACIÓN TEMPORAL:
    status = worker_ampl.getValue("solve_result")
    if status != "solved":
        _ = None
        #print(f"\n[AMPL ERROR] Estado: {state} | Status: {status} | Msg: {worker_ampl.getValue('solve_message')}")

    asignacionZ = worker_ampl.get_variable("Z").get_values().toList()
    solveResult = worker_ampl.getValue("solve_result")
    
    print(asignacionZ)

    solveMsg = worker_ampl.getValue("solve_message")

    iterations, branchNodes = getInfo(solveMsg, solver)

    # Close session to free memory
    worker_ampl.close()
    
    # Use your existing rebalance logic
    fixState = fixAssignment(asignacionZ)

    cleanFixState = tuple(int(x) for x in fixState)

    return cleanFixState, solveResult, iterations, branchNodes

def parallelLinearRelaxation(instanceContent, statesList, fixingSize, solver, max_workers=10, alphaValue=0.5, lexPoints=None, tabuList=None):
    """Parallel coordinator."""
    time0 = time()

    clean_states = []
    for state in statesList:
        clean_states.append(tuple(int(val) for val in state))
    
    # Prepare arguments for each worker
    tasks = [(instanceContent, state, fixingSize, alphaValue, lexPoints, tabuList, solver) for state in statesList]
    
    relaxStates = []
    iterations = 0
    branchNodes = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        try:
            # Dynamically execute worker tasks in isolated worker subprocesses
            results = list(executor.map(solve_single_relax_state, tasks))
            
            # FIXED: Properly unpack and check the processed results array
            for result in results:
                if result is None or result[0] is None:
                    continue
                if result[1] != "solved":
                    continue
                
                # Append individual verified points into your tracking front array
                relaxStates.append(result[0])
                iterations += result[2]
                branchNodes += result[3]
                
        except Exception as e:
            import traceback
            print(f"\nCRITICAL THREAD ERROR EXPOSED: {str(e)}")
            traceback.print_exc()
            raise e
    
    time1 = time()
    return relaxStates, time1 - time0, iterations, branchNodes

def getInfo(solveMsg, solver = "knitro"):
    iterations = 0
    branchNodes = 0
    print(f"solve MSG: {solveMsg}")
    if solveMsg:
        if solver == "gurobi":
            # Extracción para Gurobi
            matchIters = re.search(r'(\d+)\s+simplex iterations', solveMsg)
            if matchIters:
                iterations = int(matchIters.group(1))

            matchNodes = re.search(r'(\d+)\s+branching nodes', solveMsg)
            if matchNodes:
                branchNodes = int(matchNodes.group(1))

        elif solver == "knitro":
            # Extracción para Knitro
            matchIters = re.search(r'(\d+)\s+subproblem solves', solveMsg)
            if matchIters:
                iterations = int(matchIters.group(1))
            matchNodes = re.search(r'(\d+)\s+nodes', solveMsg)
            if matchNodes:
                branchNodes = int(matchNodes.group(1))

    return iterations, branchNodes