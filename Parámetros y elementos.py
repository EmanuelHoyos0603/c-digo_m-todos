# Librerías necesarias 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

# Parametros del problema

Lx, Ly = 0.08, 0.08         # Dimensiones de la placa [m] (80 mm x 80 mm)
t = 0.002                   # Espesor de la placa [m] (2 mm)
k_Cu = 385.0                # Conductividad térmica del cobre [W/(m*K)]
P = 15.0                    # Potencia disipada por el CPU [W]
T0 = 298.15                 # Temperatura en bordes refrigerados [K] (25 °C)