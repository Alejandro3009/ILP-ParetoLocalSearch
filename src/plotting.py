import matplotlib.pyplot as plt
import numpy as np
import math

def plotInstanceMap(cdList, clientList, centroids, areaSize, fileName="instance_map.png"):
    """
    Plots the physical distribution of Clients, CDs, and City Centroids.
    CD size represents Capacity, and color represents Fixed Cost.
    """
    plt.figure(figsize=(10, 8))
    
    # 1. Plot background context (optional contour for proximity)
    x = np.linspace(0, areaSize, 100)
    y = np.linspace(0, areaSize, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            dist = min([math.sqrt((X[i,j] - cx)**2 + (Y[i,j] - cy)**2) for cx, cy in centroids])
            Z[i,j] = max(0, 1 - (dist / (areaSize / 2)))
    
    plt.contourf(X, Y, Z, levels=20, cmap='YlOrRd', alpha=0.1)

    # 2. Plot Clients (Blue dots)
    plt.scatter([c.posX for c in clientList], [c.posY for c in clientList], 
                c='blue', s=15, alpha=0.5, label='Clients')

    # 3. Plot CDs (Squares: Size=Capacity, Color=FixedCost)
    # Scaling capacity for visibility (e.g., divide by 5)
    cd_scatter = plt.scatter([c.posX for c in cdList], [c.posY for c in cdList], 
                             c=[c.fixedCost for c in cdList], 
                             s=[c.capacity/5 for c in cdList], 
                             cmap='viridis', marker='s', edgecolors='black', 
                             label='CDs (Size=Cap, Color=Cost)')
    
    # 4. Plot Centroids (City Centers)
    plt.scatter([c[0] for c in centroids], [c[1] for c in centroids], 
                c='red', marker='*', s=200, edgecolors='black', label='City Centroids')

    plt.colorbar(cd_scatter, label='Fixed Cost ($)')
    plt.title('Instance Geography: Clustered Clients & Dynamic CDs')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(fileName)
    plt.show()

def plotUrbanHeatmap(centroids, areaSize, fileName="urban_heatmap.png"):
    """
    Generates a heatmap showing the Proximity Factor (Urban Intensity).
    """
    x = np.linspace(0, areaSize, 200)
    y = np.linspace(0, areaSize, 200)
    X, Y = np.meshgrid(x, y)
    
    # Calculate Proximity Factor for every point on the grid
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            minDist = min([math.sqrt((X[i,j] - cx)**2 + (Y[i,j] - cy)**2) for cx, cy in centroids])
            Z[i,j] = max(0, 1 - (minDist / (areaSize / 2)))

    plt.figure(figsize=(10, 8))
    im = plt.imshow(Z, extent=[0, areaSize, 0, areaSize], origin='lower', cmap='hot')
    plt.colorbar(im, label='Urban Intensity (Proximity Factor)')
    
    # Overlay Centroids
    plt.scatter([c[0] for c in centroids], [c[1] for c in centroids], 
                c='cyan', marker='*', s=150, edgecolors='black', label='Centroids')

    plt.title('Urban Intensity Heatmap')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.savefig(fileName)
    plt.show()