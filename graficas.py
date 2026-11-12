import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

def graficar_temperatura(coords, elementos, T, archivo_salida="mapa_de_temperatura.pdf"):
    
    # Se genera el mapa de contorno térmico.
    
    # Punto de máxima temperatura
    idx_max = np.argmax(T)
    x_max, y_max = coords[idx_max]
    t_max_k = T[idx_max]
    t_max_c = t_max_k - 273.15

    # Triangulación
    triang = mtri.Triangulation(coords[:, 0], coords[:, 1], elementos)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    # Contornos de temperatura
    niveles = np.linspace(np.min(T) - 273.15, t_max_c, 40)
    contour = ax.tricontourf(triang, T - 273.15, levels=niveles, cmap="inferno")

    # Punto rojo en la coordenada crítica para saber cual és
    ax.scatter([x_max], [y_max], color="red", s=45, edgecolors="white", linewidths=1.0, zorder=5)

    info_texto = (
        r"$T_{\max} = " f"{t_max_c:.2f}" r"^\circ\mathrm{C}\ (" f"{t_max_k:.2f}" r"\ \mathrm{K})$" "\n"
        f"Posición: ({x_max*1e3:.1f}, {y_max*1e3:.1f}) mm"
    )
    ax.text(
        0.04, 0.04, info_texto,
        transform=ax.transAxes,
        fontsize=8.5,
        verticalalignment="bottom",
        horizontalalignment="left",
        bbox=dict(boxstyle="square,pad=0.4", facecolor="white", alpha=0.85, edgecolor="#cccccc", lw=0.8),
        zorder=6
    )

    cbar = plt.colorbar(contour, ax=ax, pad=0.03)
    cbar.set_label(r"Temperatura [$^\circ\mathrm{C}$]", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Configuración de ejes
    ax.set_xlabel("x [m]", fontsize=9)
    ax.set_ylabel("y [m]", fontsize=9)
    ax.set_title("Distribución Estacionaria de Temperatura (FEM $P_1$)", fontsize=10, pad=10)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(archivo_salida, dpi=300)
    plt.show()