# Libreias necesarias
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

import parametros as par
from malla import generar_malla

# Parámetro de amplitud de prueba
DELTA_T = 10.0  # [K]

def theta_exacta(x, y, Lx, Ly):
    # Solución analítica manufacturada
    return DELTA_T * np.sin(np.pi * x / Lx) * np.sin(np.pi * y / Ly)

def fuente_manufacturada(x, y, Lx, Ly):
    # Término fuente analítico
    factor = DELTA_T * (np.pi ** 2) * (1.0 / (Lx ** 2) + 1.0 / (Ly ** 2))
    return factor * np.sin(np.pi * x / Lx) * np.sin(np.pi * y / Ly)

def resolver_mms(coords, elementos, nodos_dirichlet, Lx, Ly, k):
    # Se Ensambla y se resuelve el sistema FEM P1 usando la fuente manufacturada
    
    n_nodos = len(coords)
    filas, cols, vals_K = [], [], []
    F_global = np.zeros(n_nodos)

    pts_gauss = np.array([
        [1.0 / 6.0, 1.0 / 6.0],
        [2.0 / 3.0, 1.0 / 6.0],
        [1.0 / 6.0, 2.0 / 3.0]
    ])
    pesos_gauss = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

    for elem in elementos:
        p0, p1, p2 = coords[elem]
        x0, y0 = p0
        x1, y1 = p1
        x2, y2 = p2

        # Área del triángulo
        detJ = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        area = 0.5 * detJ

        # Matriz elemental de rigidez
        b = np.array([y1 - y2, y2 - y0, y0 - y1])
        c = np.array([x2 - x1, x0 - x2, x1 - x0])
        Ke = (k / (4.0 * area)) * (np.outer(b, b) + np.outer(c, c))

        for i in range(3):
            for j in range(3):
                filas.append(elem[i])
                cols.append(elem[j])
                vals_K.append(Ke[i, j])

        # Vector de cargas elemental F^(e)
        Fe = np.zeros(3)
        for (r, s), w in zip(pts_gauss, pesos_gauss):
            phi = np.array([1.0 - r - s, r, s])
            xg = phi[0] * x0 + phi[1] * x1 + phi[2] * x2
            yg = phi[0] * y0 + phi[1] * y1 + phi[2] * y2
            q_dot_manu = k * fuente_manufacturada(xg, yg, Lx, Ly)
            Fe += w * q_dot_manu * phi * area

        F_global[elem] += Fe

    K_global = sp.coo_matrix((vals_K, (filas, cols)), shape=(n_nodos, n_nodos)).tocsr()

    # Grados de libertad libres
    todos = np.arange(n_nodos)
    libres = np.setdiff1d(todos, nodos_dirichlet)

    theta_h = np.zeros(n_nodos)
    theta_h[libres] = spsolve(K_global[libres, :][:, libres], F_global[libres])

    return theta_h

def calcular_error_L2_y_h(coords, elementos, theta_h, Lx, Ly):
    # Calcula la norma discreta L2 del error y  h.
    pts_gauss = np.array([
        [1.0 / 6.0, 1.0 / 6.0],
        [2.0 / 3.0, 1.0 / 6.0],
        [1.0 / 6.0, 2.0 / 3.0]
    ])
    pesos_gauss = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

    error_sq = 0.0
    h_max = 0.0

    for elem in elementos:
        p0, p1, p2 = coords[elem]
        # Diámetro máximo del triángulo (arista más larga)
        l0 = np.hypot(p1[0] - p0[0], p1[1] - p0[1])
        l1 = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
        l2 = np.hypot(p0[0] - p2[0], p0[1] - p2[1])
        h_max = max(h_max, l0, l1, l2)

        detJ = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])
        area = 0.5 * detJ
        th_e = theta_h[elem]

        for (r, s), w in zip(pts_gauss, pesos_gauss):
            phi = np.array([1.0 - r - s, r, s])
            xg = phi[0] * p0[0] + phi[1] * p1[0] + phi[2] * p2[0]
            yg = phi[0] * p0[1] + phi[1] * p1[1] + phi[2] * p2[1]

            u_ex = theta_exacta(xg, yg, Lx, Ly)
            u_num = np.dot(phi, th_e)
            error_sq += w * ((u_ex - u_num) ** 2) * area

    return np.sqrt(error_sq), h_max

def ejecutar_estudio_mms():
    Lx, Ly, k = par.Lx, par.Ly, par.k_Cu
    resoluciones = [11, 21, 41, 81, 121]

    hs, errores, nodos_lista, elems_lista = [], [], [], []

    print(f"{'Malla':<10} | {'Nodos':<7} | {'Elems':<7} | {'h [m]':<10} | {'Error L2':<12} | {'Orden p':<8}")
    print("-" * 68)

    for idx, n in enumerate(resoluciones):
        # Dominio rectangular regular sin perforaciones
        coords, elementos, nodos_dirichlet = generar_malla(Lx, Ly, nx=n, ny=n, orificios=None)
        theta_h = resolver_mms(coords, elementos, nodos_dirichlet, Lx, Ly, k)
        err_L2, h_max = calcular_error_L2_y_h(coords, elementos, theta_h, Lx, Ly)

        hs.append(h_max)
        errores.append(err_L2)
        nodos_lista.append(len(coords))
        elems_lista.append(len(elementos))

        if idx == 0:
            orden_str = "---"
        else:
            p = np.log(errores[idx - 1] / errores[idx]) / np.log(hs[idx - 1] / hs[idx])
            orden_str = f"{p:.4f}"

        print(f"{f'{n}x{n}':<10} | {len(coords):<7} | {len(elementos):<7} | {h_max:<10.5f} | {err_L2:<12.4e} | {orden_str:<8}")

    # Regresión lineal en escala log-log para obtener la pendiente p
    hs = np.array(hs)
    errores = np.array(errores)
    pendiente_global, intercepto = np.polyfit(np.log(hs), np.log(errores), 1)
    print("-" * 68)
    print(f"Orden de convergencia con ajuste log-log: p = {pendiente_global:.4f} (Teórico: 2.0000)")

    # Gráfica de convergencia log-log
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.loglog(hs, errores, "o-", color="#1f77b4", linewidth=1.8, markersize=6, label=r"Error numérico $\|\theta_{\mathrm{manu}} - \theta_h\|_{L^2}$")

    # Recta de referencia teórica O(h^2)
    ref_h2 = errores[0] * (hs / hs[0]) ** 2
    ax.loglog(hs, ref_h2, "--", color="#d62728", linewidth=1.4, label=r"Referencia teórica $\mathcal{O}(h^2)$")

    ax.set_xlabel(r"Tamaño de malla $h$ [m]", fontsize=9.5)
    ax.set_ylabel(r"Error en norma $L^2(\Omega)$ [$\mathrm{K}\cdot\mathrm{m}$]", fontsize=9.5)
    ax.set_title(f"Verificación MMS: Convergencia cuadrática ($p = {pendiente_global:.2f}$)", fontsize=10.5, pad=10)
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend(fontsize=8.5, framealpha=0.9)
    ax.tick_params(labelsize=8.5)

    plt.tight_layout()
    plt.savefig("convergencia_mms.pdf", dpi=300)
    plt.savefig("convergencia_mms.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    ejecutar_estudio_mms()