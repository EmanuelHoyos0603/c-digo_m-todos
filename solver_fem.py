# Librerias necesarias
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

def matrices_elementales(coords_elem, k, q_val):
    # Se Calcula Ke y Fe para un triángulo lineal P1.
    x0, y0 = coords_elem[0]
    x1, y1 = coords_elem[1]
    x2, y2 = coords_elem[2]

    detJ = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    area = 0.5 * detJ

    b = np.array([y1 - y2, y2 - y0, y0 - y1])
    c = np.array([x2 - x1, x0 - x2, x1 - x0])

    Ke = (k / (4.0 * area)) * (np.outer(b, b) + np.outer(c, c))
    Fe = (area * q_val / 3.0) * np.ones(3)
    return Ke, Fe

def ensamblar_sistema(coords, elementos, k, q_dot, bounds_cpu):
    # Se ensambla la matriz global Ky el vector F.
    num_nodos = len(coords)
    x1, x2, y1, y2 = bounds_cpu

    filas, cols, datos = [], [], []
    F = np.zeros(num_nodos)

    for elem in elementos:
        coords_elem = coords[elem]
        xc, yc = np.mean(coords_elem, axis=0)
        q_val = q_dot if (x1 <= xc <= x2 and y1 <= yc <= y2) else 0.0

        Ke, Fe = matrices_elementales(coords_elem, k, q_val)

        for i_loc in range(3):
            i_glob = elem[i_loc]
            F[i_glob] += Fe[i_loc]
            for j_loc in range(3):
                j_glob = elem[j_loc]
                filas.append(i_glob)
                cols.append(j_glob)
                datos.append(Ke[i_loc, j_loc])

    K = coo_matrix((datos, (filas, cols)), shape=(num_nodos, num_nodos)).tocsr()
    return K, F

def resolver_fem(K, F, nodos_dirichlet, T0):
    # Se resuelve K_uu * theta_u = F_u teniendo en cuenta la variable de exceso térmico
    num_nodos = K.shape[0]
    nodos_libres = np.setdiff1d(np.arange(num_nodos), nodos_dirichlet)

    K_uu = K[nodos_libres, :][:, nodos_libres]
    F_u = F[nodos_libres]

    theta_u = spsolve(K_uu, F_u)

    theta = np.zeros(num_nodos)
    theta[nodos_libres] = theta_u
    return theta + T0