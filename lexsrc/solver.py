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
    ampl.option["gurobi_options"] = "NonConvex=2 MIPGap=1e-8 FeasTol=1e-9 BarConvTol=1e-9 timelimit=3600"
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
    ampl.option["gurobi_options"] = "NonConvex=2 MIPGap=1e-8 FeasTol=1e-9 BarConvTol=1e-9 timelimit=1800"
    
    ampl.solve()
    
    solveResult = ampl.getValue("solve_result")
    if solveResult == "solved":
        transp = ampl.getValue("CostoTransp")
        infra = ampl.getValue("CostoInfra")
        return transp, infra
    return None, None

# Filter out dominated points from the epsilon front
def filterEpsilonFront(points):
    filtered = []
    for p1 in points:
        isDominated = False
        for p2 in points:
            # Strong dominance check: p1 is worse than p2 in both objectives
            if p1[0] >= p2[0] and p1[1] >= p2[1] and p1 != p2:
                isDominated = True
                break
        if not isDominated:
            filtered.append(p1)
    # Sort by Infrastructure (Y) to ensure a continuous line plot
    return sorted(filtered, key=lambda p: p[1])