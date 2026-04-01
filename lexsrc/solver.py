import time
from amplpy import AMPL, ampl_notebook

workerAmpl = None

def solveInstance(instance, model, epsilon = 1e20):
    ampl = AMPL()
    ampl.eval("reset;")
    ampl.eval(model)
    ampl.eval(instance)
    ampl.param["epsilon"] = epsilon
    ampl.setOption("solver", "gurobi")
    ampl.option["gurobi_options"] = "NonConvex=2 MIPGap=1e-8 FeasTol=1e-9 BarConvTol=1e-9"
    ampl.solve()
    
    transp = ampl.getValue("CostoTransp")
    infra = ampl.getValue("CostoInfra")
    print(f"Cds abiertos: {ampl.getData("Z")} ")

    return transp, infra

def solveEpsilon(instance, model, epsilonValue):
    ampl = AMPL()
    ampl.eval("reset;")
    ampl.eval(model)
    ampl.eval(instance)
    ampl.param["epsilon"] = epsilonValue
    ampl.setOption("solver", "gurobi")
    
    ampl.setOption("gurobi_options", "outlev=0") 
    ampl.option["gurobi_options"] = "NonConvex=2 MIPGap=0.05"
    
    ampl.solve()
    
    solveResult = ampl.getValue("solve_result")
    if solveResult == "solved":
        transp = ampl.getValue("CostoTransp")
        infra = ampl.getValue("CostoInfra")
        return transp, infra
    return None, None


##################Funciones para la paralelización (No usadas ahora mismo)######################
def initWorker(modelFile, dataFile, licenseUuid, gurobiOptions):
    global workerAmpl

    workerAmpl = ampl_notebook(modules=["gurobi"], license_uuid=licenseUuid)

    workerAmpl.read(modelFile)
    workerAmpl.readData(dataFile)
    
    workerAmpl.setOption("solver_msg", 0)
    
    workerAmpl.setOption("solver", "gurobi")
    
    full_gurobi_opts = f"{gurobiOptions} outlev=0"
    workerAmpl.option["gurobi_options"] = full_gurobi_opts
    
    workerAmpl.setOption("os_options", "outlev=0")
    

def solveWorker(chromosome):
    """Resuelve un individuo."""
    global workerAmpl
    
    if workerAmpl is None:
        return float('inf'), 0.0
        
    if sum(chromosome) == 0:
        return float('inf'), 0.0
        
    try:
        # Fijar variables Z
        fixCommands = [f"fix Z[{i}] := {x};" for i, x in enumerate(chromosome)]
        workerAmpl.eval("".join(fixCommands))
        
        t0 = time.time()
        workerAmpl.solve()
        t1 = time.time()
        
        solveResult = workerAmpl.getValue("solve_result")
        
        if solveResult == "solved" or solveResult == "limit":
            objValue = workerAmpl.getObjective("TotalCost").value()
        else:
            objValue = float('inf')
            
        return objValue, (t1 - t0)
        
    except Exception as e:
        return float('inf'), 0.0