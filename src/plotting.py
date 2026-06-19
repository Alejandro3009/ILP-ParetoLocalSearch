import matplotlib.pyplot as plt
import numpy as np
import math

def plotParetoFront(heuristicFront, epsilonFront, initialSolutions, imageName, generationMethod):
        # Extract objective values from the paretoPoint instances 
        # objValueX = Infrastructure Cost, objValueY = Transport Cost
        infra_costs = [p.Infrastructure for p in heuristicFront]
        trans_costs = [p.Transport for p in heuristicFront]

        # Create the plot
        plt.figure(figsize=(10, 6))
        
        # Plot the lexicographic points for reference
        # In main.py, change the order to (Infra, Transport)
        plt.scatter([epsilonFront['transMin'], epsilonFront['transMax']], [epsilonFront['infraMax'], epsilonFront['infraMin']], c=['blue', 'red'])
        plt.plot(epsilonFront['paretoX'], epsilonFront['paretoY'], marker='o', linestyle='-', color='green', label='Epsilon Frontier') 

        # Plot individual points
        plt.scatter(trans_costs, infra_costs, color='purple', zorder=5, label='Pareto Optimal Points')

        # Optional: Draw a line connecting the points to visualize the 'Frontier'
        # We sort by Infrastructure Cost to ensure the line connects points in order
        sorted_front = sorted(heuristicFront, key=lambda p: p.Infrastructure)
        x_line = [p.Infrastructure for p in sorted_front]
        y_line = [p.Transport for p in sorted_front]
        plt.plot(y_line, x_line, color='blue', linestyle='--', alpha=0.6, label='Pareto Frontier')

        # Highlight the initial solution
        initialInfra = [p.Infrastructure for p in initialSolutions]
        initialTrans = [p.Transport for p in initialSolutions]
        plt.scatter(initialTrans, initialInfra, color='orange', marker='X', s=100, label='Initial Solution')

        # Labels and Titles
        plt.title(f"Results for {imageName}")
        plt.xlabel('Transport Cost ($)')
        plt.ylabel('Infrastructure Cost ($)')
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend()

        # Save the plot
        plt.savefig(f"{imageName}_using_{generationMethod}_results.png")
        #plt.show()
        print(f"Visualisation saved as '{imageName}_using_{generationMethod}_results.png'")