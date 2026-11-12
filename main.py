# Librerias necesarias
import parametros as par
from malla import generar_malla
from solver_fem import ensamblar_sistema, resolver_fem
from graficas import graficar_temperatura

def main():
    # Relacionado con la malla
    coords, elementos, nodos_dirichlet = generar_malla(
        par.Lx, par.Ly, nx=61, ny=61, orificios=par.orificios
    )
    print(f"Malla generada: {len(coords)} nodos, {len(elementos)} elementos.")

    # Relacioando con el ensamble del sistema
    bounds_cpu = (par.x1_cpu, par.x2_cpu, par.y1_cpu, par.y2_cpu)
    K, F = ensamblar_sistema(coords, elementos, par.k_Cu, par.q_dot, bounds_cpu)

    # Relacionado con la solución
    T = resolver_fem(K, F, nodos_dirichlet, par.T0)
    print(f"Temperatura máxima: {T.max():.2f} K ({T.max() - 273.15:.2f} °C)")

    # Relacionado con las gráficas
    graficar_temperatura(coords, elementos, T)

if __name__ == "__main__":
    main()