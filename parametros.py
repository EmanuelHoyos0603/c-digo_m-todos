# Parámetros físicos del disipador de calor (placa de cobre) y del CPU
Lx, Ly = 0.08, 0.08         # Dimensiones de la placa [m] (80 mm x 80 mm)
t = 0.002                   # Espesor de la placa [m] (2 mm)
k_Cu = 385.0                # Conductividad térmica del cobre [W/(m*K)]
P = 15.0                    # Potencia disipada por el CPU [W]
T0 = 298.15                 # Temperatura en bordes refrigerados [K] (25 °C)


# Localización del CPU en el centro de la placa
x1_cpu, x2_cpu = 0.03, 0.05 # [m] (20 mm x 20 mm centrada)
y1_cpu, y2_cpu = 0.03, 0.05 # [m] (20 mm x 20 mm centrada)
A_cpu = (x2_cpu - x1_cpu) * (y2_cpu - y1_cpu) #
q_dot = P / (A_cpu * t)     # Generación volumétrica equivalente [W/m^3]

# Coordenadas y radio de los 4 orificios de sujeción (x, y, r) [m]
orificios = [
    (0.015, 0.015, 0.003),
    (0.065, 0.015, 0.003),
    (0.015, 0.065, 0.003),
    (0.065, 0.065, 0.003)
]