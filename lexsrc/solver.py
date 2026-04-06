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