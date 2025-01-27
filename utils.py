import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np


def validatePoses(bodyPoints, exercise):
    if (exercise == "ts1"):
        return validatePushUps(bodyPoints)
    elif (exercise == "ts2"):
        return validatePullUps(bodyPoints)
    elif (exercise == "ts3"):
        return validateShoulderPress(bodyPoints)
    elif exercise == "ts4":
        return validateRowing(bodyPoints)

def plot_regression(angles, output_path='regression_plot.png'):
    """
    Grafica una regresión lineal para una lista de ángulos y guarda la imagen.

    Parámetros:
    angles (list of float): Lista de ángulos medidos en cada fotograma o tiempo.
    output_path (str): Ruta del archivo para guardar el gráfico.
    """
    # Genera los datos del eje X (marcos de tiempo)
    x = np.arange(len(angles)).reshape(-1, 1)
    y = np.array(angles)

    # Crea y ajusta el modelo de regresión lineal
    model = LinearRegression()
    model.fit(x, y)

    # Predice los valores de y usando la regresión
    y_pred = model.predict(x)

    # Grafica los ángulos originales y la línea de regresión
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='Ángulos detectados', color='blue', marker='o')
    plt.plot(x, y_pred, label='Línea de regresión', color='red', linestyle='--')

    # Etiquetas y título
    plt.xlabel('Fotograma')
    plt.ylabel('Ángulo (grados)')
    plt.title('Ángulos durante el Ejercicio')
    plt.legend()
    plt.grid(True)

    # Guarda el gráfico en el archivo especificado
    plt.savefig(output_path)
    print(f'Gráfico guardado en: {output_path}')

def calculateAngle(a, b, c):
    """
    Calcula el ángulo entre tres puntos: a (inicio), b (vértice), c (fin).
    """
    ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    return abs(ang if ang >= 0 else ang + 360)

def validatePushUps(bodyPoints):
    """
    Calcula las repeticiones de flexiones (push-ups) basándose en los puntos del cuerpo.
    
    Args:
        bodyPoints (list): Lista de puntos clave del cuerpo en cada fotograma,
                           donde cada entrada es un diccionario con las posiciones
                           de las articulaciones en coordenadas normalizadas.

    Returns:
        int: Número total de repeticiones completadas correctamente.
    """
    repetitions = 0
    is_down = False

    for points in bodyPoints:
        if all(k in points for k in ["RShoulder", "RElbow", "RWrist", "RHip", "RKnee"]):
            RShoulder = points["RShoulder"]
            RElbow = points["RElbow"]
            RWrist = points["RWrist"]

            angle = calculateAngle(RShoulder, RElbow, RWrist)

            if angle < 95: 
                is_down = True
            elif angle > 160 and angle < 190 and is_down:
                repetitions += 1
                is_down = False 

    return {
        "is_Valid": repetitions >= 7,
        "message": f"Haz hecho {repetitions} repeticiones correctamente."
    }

def validatePullUps(bodyPoints):
    repetitions = 0
    correctRepetitions = 0
    movement_phase = "initial"
    highest = 0
    lowest = 0
    tolerance = 20
    downPositionValid = False
    upPositionValid = False

    for points in bodyPoints:
        currentPosition = points['Nose'][1]

        RElbow = points["RElbow"]
        LElbow = points["LElbow"]
        RShoulder = points["RShoulder"]
        LShoulder = points["LShoulder"]
        RHip = points["RHip"]
        LHip = points["LHip"]

        rAngle = calculateAngle(RElbow, RShoulder, RHip)
        lAngle = calculateAngle(LHip, LShoulder, LElbow)

        if movement_phase == "initial":
            highest = currentPosition - tolerance
            lowest = currentPosition
            movement_phase = "up"
        
        if currentPosition < highest and movement_phase == "initial":
            movement_phase = "down"
        
        if currentPosition < lowest:
            lowest = currentPosition 
        
        if currentPosition > lowest + tolerance and movement_phase == "down":
            movement_phase = "up"
            highest = currentPosition
            repetitions += 1
            
            if rAngle > 141 and rAngle < 157 and lAngle > 140 and lAngle < 157:
                downPositionValid = True

            if downPositionValid and upPositionValid:
                correctRepetitions += 1
                downPositionValid = False
                upPositionValid = False
        
        if currentPosition > highest:
            highest = currentPosition

        if currentPosition < highest - tolerance and movement_phase == "up":
            movement_phase = "down"
            lowest = currentPosition

            if rAngle > 30 and rAngle < 50 and lAngle > 30 and lAngle < 50:
                upPositionValid = True

    return {
        "is_Valid": correctRepetitions / repetitions >= 0.7,
        "message": f"Haz hecho {correctRepetitions} repeticiones correctamente."
    }
    
def validateShoulderPress(bodyPoints):
    repetitions = 0
    is_down = False

    for points in bodyPoints:
        if all(k in points for k in ["RShoulder", "RElbow", "RWrist"]):
            RShoulder = points["RShoulder"]
            RElbow = points["RElbow"]
            RWrist = points["RWrist"]

            angle = calculateAngle(RShoulder, RElbow, RWrist)

            if angle < 95: 
                is_down = True
            elif angle > 160 and angle < 190 and is_down:
                repetitions += 1
                is_down = False 

    return {
        "is_Valid": repetitions >= 7,
        "message": f"Haz hecho {repetitions} repeticiones correctamente."
    }

def validateRowing(bodyPoints):
    repetitions = 0
    is_down = False

    for points in bodyPoints:
        LShoulder = points["LShoulder"]
        LElbow = points["LElbow"]
        LWrist = points["LWrist"]

        angle = calculateAngle(LShoulder, LElbow, LWrist)
        if angle < 95: 
            is_down = True
        elif angle > 160 and angle < 190 and is_down:
            repetitions += 1
            is_down = False

    return {
        "is_Valid": repetitions >= 7,
        "message": f"Haz hecho {repetitions} repeticiones correctamente."
    }
