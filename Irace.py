# irace_launcher.py
import sys
import argparse
import os  # Added to manage directory contexts dynamically
from time import time
# Import your existing pipeline blocks
from src.utils import loadDatInstance, calcularHipervolumen, invertedGenerationalDistance, loadConfig
from src.initialSolution import generateInitialSolution
from src.solver import calculateFitnessParallel
from src.TPLS_MPS import MultiPointParetoSearch
from src.TPLS_OPS import onePointParetoSearch
from lexsrc.utils import loadEpsilonResults

def main():
    parser = argparse.ArgumentParser(description="irace Hook for TPLS Tuning")
    
    # 1. Baseline parameters
    parser.add_argument('--instance', type=str, required=True)
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--movementSize', type=int, default=3)
    parser.add_argument('--tabuListSize', type=int, default=5)
    parser.add_argument('--tabuTenure', type=int, default=5)
    parser.add_argument('--neiborsGenerated', type=int, default=10)
    
    # 2. Add the new parameters you defined in parameters.txt
    parser.add_argument('--amountInitialSolutions', type=int, default=5)
    parser.add_argument('--initialization', type=str, default="random")
    parser.add_argument('--searchMethod', type=str, default="steepestDescent")
    parser.add_argument('--neiborhoodGeneration', type=str, default="random")
    
    args = parser.parse_args()

    # Dynamic path routing: Finds the absolute path to config.json at the project root
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "config.json")

    # Load shared baseline configurations using the secure path
    configData = loadConfig(config_path)
    instanceName = args.instance
    
    currentInstance = loadDatInstance(instanceName)
    previousResults = {
        "transMin": 1128159000.0,
        "transMax": 1260880000.0,
        "infraMin": 2092589.5887076436,
        "infraMax": 2929909.823477363,
        "paretoX": [1128159000, 1132231000, 1135079000, 1142801000, 1145270000, 1148633000, 1148633000, 1148633000, 1155756000, 1171020000, 1186155000, 1201443000, 1208486000, 1216278000, 1228201000, 1234979000, 1260880000, 1260880000],
        "paretoY": [2929909.823477363, 2729547.7304013832, 2642004.1295820596, 2617521.0343230367, 2557767.496913044, 2273609.935810578, 2371696.333630884, 2305465.9746205104, 2270006.2643962973, 2247268.852768442, 2244754.906391519, 2207705.9602863365, 2185625.170348724, 2149997.5252366727, 2140468.707225944, 2140026.067883053, 2092589.5887076433, 2092546.460677789]
    }
    
    # Run setup blocks identically to main.py
    initialState = generateInitialSolution(args.initialization, currentInstance)

    #print(f"DEBUG_INIT - Estado inicial sanitizado: {initialState}")

    if isinstance(initialState, list):
        initialState = [tuple(int(val) for val in sub_tup) for sub_tup in initialState]
    else:
        initialState = [tuple(int(val) for val in initialState)]

    #print(f"DEBUG_INIT - Estado inicial sanitizado: {initialState}")
    
    # Run Multi-Point Tabu Search using tuned parameters
    # Inside Irace.py main execution block
    
    # Check what algorithm strategy irace wants to test for this configuration
    if args.searchMethod == "firstDescent":
        # If irace chooses firstDescent, call your local search module
        finalParetoFront, _, _, _, _ = onePointParetoSearch(
            initialState, 
            currentInstance,
            args.neiborhoodGeneration,  # Passed to 'movementOperator' positional slot
            None,  # Passed to 'lexPoints' positional slot
            neiborsGenerated=args.neiborsGenerated,
            iterationAmount=args.iterations,
            maxIterationsWithoutImprovement=configData['plsParams']['maxIterationsWithoutImprovement'],
            movementSize=args.movementSize,
            tabuListSize=args.tabuListSize,
            tabuTenure=args.tabuTenure,
            alpha=configData['plsParams']['alpha'],
            maxWorkers=configData['plsParams']['maxWorkers']
        )
    else:
        # If irace chooses steepestDescent, call your multi-point module
        finalParetoFront, _, _, _, _ = MultiPointParetoSearch(
            initialState, 
            currentInstance,
            args.neiborhoodGeneration,
            [previousResults['transMax'], previousResults['infraMax']],
            iterationAmount=args.iterations,
            neiborsGenerated=args.neiborsGenerated,
            maxIterationsWithoutImprovement=configData['plsParams']['maxIterationsWithoutImprovement'],
            movementSize=args.movementSize,
            tabuListSize=args.tabuListSize,
            tabuTenure=args.tabuTenure,
            alpha=configData['plsParams']['alpha'],
            maxWorkers=configData['plsParams']['maxWorkers']
        )

    # Temporary diagnostic print inside Irace.py
    #print(f"DEBUG_FRONT - Longitud del frente encontrado: {len(finalParetoFront) if finalParetoFront else 0}")
    if finalParetoFront:
        _ = None
        #print(f"DEBUG_FRONT - Puntos: {[p.state for p in finalParetoFront]}")

    if not finalParetoFront:
        print("999.0") # Return a terrible cost if it fails
        return
    
    # Calculate metrics for irace to minimize
    # Calculate Hypervolume
    hvValue = calcularHipervolumen(
        finalParetoFront, 
        previousResults['transMin'], previousResults['transMax'], 
        previousResults['infraMin'], previousResults['infraMax']
    )
    
    # Calculate Inverted Generational Distance
    igdValue = invertedGenerationalDistance(
        previousResults['paretoX'], previousResults['paretoY'], 
        finalParetoFront, 
        [previousResults['infraMin'], previousResults['infraMax']], 
        [previousResults['transMin'], previousResults['transMax']]
    )
    
    # 1. Print Hypervolume first (prefixed so you can track it in console logs)
    #sys.stderr.write(f"[MONITOR] Iteracion Completada - Hypervolume: {hvValue:.6f}\n")
    
    # 2. CRITICAL: irace reads ONLY the LAST line printed to standard output.
    # This must remain the objective value irace is actively minimizing (IGD).
    print(f"{igdValue:.6f}")

if __name__ == "__main__":
    main()