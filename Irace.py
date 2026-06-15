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
    parser.add_argument('--relaxedGeneration', type=str, default="True")
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
    # Inside your Irace.py main() function

    # Hardcoded reference bounds extracted from fronts/20x10-3.json
    # This prevents you from changing any shared utils or paths!
    instance_bounds = {
        "20x10-3": {
            "infraMin": 2092589.5887076436,   # Replace these with your actual exact values
            "infraMax": 2929909.823477363,  # from your fronts/20x10-3.json file
            "transMin": 1128159000.0,
            "transMax": 1260880000.0,
            # Paste your exact epsilon points here for perfect IGD tracking:
            "paretoX": [
                1128159000,
                1132231000,
                1135079000,
                1142801000,
                1145270000,
                1148633000,
                1148633000,
                1148633000,
                1155756000,
                1171020000,
                1186155000,
                1201443000,
                1208486000,
                1216278000,
                1228201000,
                1234979000,
                1260880000,
                1260880000
            ], 
            "paretoY": [
                2929909.823477363,
                2729547.7304013832,
                2642004.1295820596,
                2617521.0343230367,
                2557767.496913044,
                2273609.935810578,
                2371696.333630884,
                2305465.9746205104,
                2270006.2643962973,
                2247268.852768442,
                2244754.906391519,
                2207705.9602863365,
                2185625.170348724,
                2149997.5252366727,
                2140468.707225944,
                2140026.067883053,
                2092589.5887076433,
                2092546.460677789
            ]
        }
    }

    # Safely extract instance metadata without using relative file search paths
    instance_key = args.instance.split('/')[-1].replace('.dat', '')
    bounds = instance_bounds.get(instance_key, instance_bounds["20x10-3"])

    # Now pass these clean static boundaries into your execution blocks
    initialState = generateInitialSolution(args.initialization, currentInstance)

    finalParetoFront, _, _ = MultiPointParetoSearch(
        initialState, 
        currentInstance,
        args.neiborhoodGeneration,
        [bounds['transMax'], bounds['infraMax']], # Exact numeric bounds passed to Gurobi!
        iterationAmount=args.iterations, 
        neiborsGenerated=args.neiborsGenerated,
        maxIterationsWithoutImprovement=configData['plsParams']['maxIterationsWithoutImprovement'],
        movementSize=args.movementSize, 
        tabuListSize=args.tabuListSize, 
        tabuTenure=args.tabuTenure, 
        alpha=configData['plsParams']['alpha'],
        maxWorkers=configData['plsParams']['maxWorkers']
    )

    if not finalParetoFront:
        print("999.0") # Return a terrible cost if it fails
        return

    # Calculate metrics for irace to minimize
    # Calculate Hypervolume
    hvValue = calcularHipervolumen(
        finalParetoFront, 
        bounds['transMin'], bounds['transMax'], 
        bounds['infraMin'], bounds['infraMax']
    )
    
    # Calculate Inverted Generational Distance
    igdValue = invertedGenerationalDistance(
        bounds['paretoX'], bounds['paretoY'], 
        finalParetoFront, 
        [bounds['infraMin'], bounds['infraMax']], 
        [bounds['transMin'], bounds['transMax']]
    )
    
    # 1. Print Hypervolume first (prefixed so you can track it in console logs)
    print(f"DEBUG_METRIC - Hypervolume: {hvValue:.6f}")
    
    # 2. CRITICAL: irace reads ONLY the LAST line printed to standard output.
    # This must remain the objective value irace is actively minimizing (IGD).
    print(f"{igdValue:.6f}")

if __name__ == "__main__":
    main()