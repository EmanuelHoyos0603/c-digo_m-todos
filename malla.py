import numpy as np
from scipy.spatial import Delaunay

def generar_malla(Lx, Ly, nx=61, ny=61, orificios=None):
    
    # Se genera una malla triangular P1 no estructurada
    
    # Se crea una nube base de puntos en el dominio rectangular
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    xx, yy = np.meshgrid(x, y)
    coords = np.column_stack([xx.ravel(), yy.ravel()])

    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    h_prom = 0.5 * (dx + dy)

    if orificios:
        for (hx, hy, r) in orificios:
            # Se eliminan puntos base que caen dentro o muy cerca del borde del círculo
            dist = np.hypot(coords[:, 0] - hx, coords[:, 1] - hy)
            coords = coords[dist > r + 0.55 * h_prom]

            # Se colocan nodos exactamente sobre la circunferencia de radio r
            n_circ = max(24, int(np.ceil(2 * np.pi * r / (0.6 * h_prom))))
            angulos = np.linspace(0, 2 * np.pi, n_circ, endpoint=False)
            x_circ = hx + r * np.cos(angulos)
            y_circ = hy + r * np.sin(angulos)
            coords = np.vstack([coords, np.column_stack([x_circ, y_circ])])

    # 2. Triangulación de Delaunay sobre la nube de puntos ajustada (suavisar perforaciones)
    tri = Delaunay(coords)
    elementos = tri.simplices

    # Se filtran los triángulos cuyo baricentro cae dentro de los orificios
    if orificios:
        centroides = np.mean(coords[elementos], axis=1)
        mascara_valida = np.ones(len(elementos), dtype=bool)
        for (hx, hy, r) in orificios:
            dist_c = np.hypot(centroides[:, 0] - hx, centroides[:, 1] - hy)
            mascara_valida &= (dist_c >= r - 1e-9)
        elementos = elementos[mascara_valida]

    # Se asegura la orientación antihoraria en todos los triángulos (detJ > 0)
    for i, elem in enumerate(elementos):
        p0, p1, p2 = coords[elem]
        detJ = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])
        if detJ < 0:
            elementos[i] = [elem[0], elem[2], elem[1]]

    # Se identifican nodos del contorno exterior (Dirichlet)
    tol = 1e-6
    nodos_dirichlet = np.where(
        (coords[:, 0] <= tol) | (coords[:, 0] >= Lx - tol) |
        (coords[:, 1] <= tol) | (coords[:, 1] >= Ly - tol)
    )[0]

    return coords, elementos, nodos_dirichlet