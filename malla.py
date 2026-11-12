# Librerias necesarias
import numpy as np

# Funcion para generar la malla
# nx es el numero de nodos en x, ny es el numero de nodos en y
def generar_malla(Lx, Ly, nx=61, ny=61, orificios=None):

    # La idea es generar una malla triangular P1 e identificar los nodos de frontera Dirichlet
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    xx, yy = np.meshgrid(x, y) # arrays 2D de coordenadas de los nodos de la malla
    coords = np.column_stack([xx.ravel(), yy.ravel()]) # Lo que se hace construir una lista de todos los nodos de la malla, donde cada fila es un punto (x, y).

    elementos = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            n0 = j * nx + i 
            n1 = n0 + 1
            n2 = (j + 1) * nx + i
            n3 = n2 + 1
            # Se especifica la orientación antihoraria
            elementos.append([n0, n1, n3]) # Se genera el primer triángulo de la celda
            elementos.append([n0, n3, n2]) # Se genera el segundo triángulo de la celda
    elementos = np.array(elementos)

    # Remover orificios internos (frontera Neumann)
    if orificios:
        nodos_validos = np.ones(len(coords), dtype=bool)
        for (hx, hy, r) in orificios:
            dist = np.hypot(coords[:, 0] - hx, coords[:, 1] - hy)
            nodos_validos &= (dist >= r)

        mascara_elem = np.all(nodos_validos[elementos], axis=1)
        elementos = elementos[mascara_elem]

        # Reindexar los nodos activos
        nodos_unicos, nuevos_indices = np.unique(elementos, return_inverse=True)
        elementos = nuevos_indices.reshape(elementos.shape)
        coords = coords[nodos_unicos]

    # Detectar bordes exteriores (Dirichlet)
    tol = 1e-6
    nodos_dirichlet = np.where(
        (coords[:, 0] <= tol) | (coords[:, 0] >= Lx - tol) |
        (coords[:, 1] <= tol) | (coords[:, 1] >= Ly - tol)
    )[0]

    return coords, elementos, nodos_dirichlet