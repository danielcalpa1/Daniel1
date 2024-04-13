import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd
import simpy

# Definición de la clase Vector, la funcion arctan en series de Taylor y la clase ProjectileSimulator
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)

def arctan_taylor(x, n):
    result = 0
    sign = 1
    for i in range(1, n+1, 2):
        result += sign * (x**i) / i
        sign *= -1
    return result

class ProjectileSimulator:
    def __init__(self, initial_position, initial_velocity, gravity):
        self.position = initial_position
        self.velocity = initial_velocity
        self.gravity = gravity
        self.positions = []

    def update(self, time_step):
        gravity_force = Vector(0, -self.gravity)
        delta_velocity = gravity_force * (time_step / 2)
        self.velocity = self.velocity + delta_velocity
        displacement = self.velocity * time_step
        self.position = self.position + displacement
        self.positions.append((self.position.x, self.position.y))
        return self.position

# Solicitar al usuario las opciones
opcion = input("Elige una opción para solucionar tu problema de movimiento parabólico 1:Forma Escalar 2:Forma Vectorial: ")

if opcion == "1":
    # Solicitar al usuario los parámetros del problema
    while True:
        try:
            initial_position = float(input("Ingrese la posición inicial: "))
            break
        except ValueError:
            print("Error: Ingrese un valor numérico para la posición inicial.")

    initial_velocity = float(input("Ingrese la velocidad inicial: "))
    damping_coefficient = float(input("Ingrese el coeficiente de amortiguamiento: "))

    # Definir la función de la ecuación diferencial para la posición
    def motion_equation(t, y, initial_velocity, damping_coefficient):
        return [y[1], initial_velocity * np.exp(-damping_coefficient * t) - damping_coefficient * y[0]]

    # Función para detectar eventos (intersección con el eje x)
    def motion_event(t, y, initial_velocity, damping_coefficient):
        return y[0]

    # Parámetros del problema
    initial_conditions = [initial_position, initial_velocity]

    # Resolver la ecuación de movimiento utilizando scipy
    event = solve_ivp(motion_equation, [0, 10], initial_conditions, args=(initial_velocity, damping_coefficient), events=lambda t, y, initial_velocity=initial_velocity, damping_coefficient=damping_coefficient: motion_event(t, y, initial_velocity, damping_coefficient), t_eval=np.linspace(0, 10, 100))

    # Obtener el tiempo de la intersección con el eje x
    intersection_time = event.t_events[0][-1]

    # Resolver la ecuación de movimiento hasta el tiempo de intersección
    solution = solve_ivp(motion_equation, [0, intersection_time], initial_conditions, args=(initial_velocity, damping_coefficient), t_eval=np.linspace(0, intersection_time, 100))

    # Visualizar la posición en función del tiempo
    plt.figure(figsize=(10, 6))
    plt.plot(solution.t, solution.y[0])
    plt.title('Posición en Función del Tiempo')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Posición (m)')
    plt.grid()
    plt.show()

    # Convertir la solución en un DataFrame de Pandas
    data = pd.DataFrame({'Tiempo': solution.t, 'Posición': solution.y[0]})
    print(data)

elif opcion == "2":
    # Solicitar al usuario las coordenadas iniciales
    initial_position_x = eval(input("Ingrese la coordenada x de la posición inicial: "))
    initial_position_y = eval(input("Ingrese la coordenada y de la posición inicial: "))

    initial_velocity_x = eval(input("Ingrese la coordenada x de la velocidad inicial: "))
    initial_velocity_y = eval(input("Ingrese la coordenada y de la velocidad inicial: "))
    
    # Crear objetos Vector para la posición y velocidad inicial
    initial_position = Vector(initial_position_x, initial_position_y)
    initial_velocity = Vector(initial_velocity_x, initial_velocity_y)

    gravity = 9.8  # Aceleración debida a la gravedad en m/s^2

    # Crear el simulador con los valores iniciales
    simulator = ProjectileSimulator(initial_position, initial_velocity, gravity)

    time_step = 0.1  # Intervalo de tiempo para cada actualización en segundos
    
    # Crear un entorno de simulación
    env = simpy.Environment()

    # Definir una función para actualizar la posición del proyectil en cada intervalo de tiempo
    def update_position(env, simulator, time_step):
      while True:
        current_position = simulator.update(time_step)
        if current_position.y < 0:
            break
        yield env.timeout(time_step)

    # Iniciar el proceso de actualización de posición en el entorno de simulación
    env.process(update_position(env, simulator, time_step))
    
    # Iniciar la simulación
    env.run()
    
    # Extraer posiciones
    x_positions = [pos[0] for pos in simulator.positions]
    y_positions = [pos[1] for pos in simulator.positions]

    # Calcular el alcance horizontal y vertical
    horizontal_range = max(x_positions) - min(x_positions)
    vertical_range = max(y_positions) - min(y_positions)

    print(f"Alcance horizontal: {horizontal_range:.2f} metros")
    print(f"Alcance vertical: {vertical_range:.2f} metros")

     # Calcular el ángulo de tiro
    angle_rad = arctan_taylor(initial_velocity_y / initial_velocity_x, 10)
    angle_deg = np.rad2deg(angle_rad)
    print(f"Ángulo de tiro: {angle_deg:.2f} grados")

    # Graficar las posiciones
    plt.plot(x_positions, y_positions)
    plt.title("Simulación de Movimiento Parabólico")
    plt.xlabel("Posición en x (metros)")
    plt.ylabel("Posición en y (metros)")
    plt.grid(True)
    plt.show()

else:
    print("Opción no válida")
