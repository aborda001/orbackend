import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np


def validatePoses(bodyPoints, exercise):
    if (exercise == "ts1"):
        return calculateAnglePushUps(bodyPoints)
    elif (exercise == "ts2"):
        return calculateAnglesPullUps(bodyPoints)
    elif (exercise == "ts3"):
        return calculateAngleShoulderPress(bodyPoints)

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
    """Calcula el ángulo entre tres puntos a, b y c."""
    radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    angle = abs(radians * 180.0 / math.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


def calculateAnglePushUps(bodyPoints):
    anglesRight = []
    anglesLeft = []
    for points in bodyPoints:
        RShoulder = points["RShoulder"]
        RHip = points["RHip"]
        RAnkle = points["RAnkle"]
        LShoulder = points["LShoulder"]
        LHip = points["LHip"]
        LAnkle = points["LAnkle"]

        angleRight = calculateAngle(RShoulder, RHip, RAnkle)
        angleLeft = calculateAngle(LShoulder, LHip, LAnkle)
        
        anglesRight.append(angleRight)
        anglesLeft.append(angleLeft)

    plot_regression(anglesRight, 'regression_plot_right.png')
    plot_regression(anglesLeft, 'regression_plot_left.png')

    sumRight = sum(anglesRight) 
    sumLeft = sum(anglesLeft)

    averageRight = sumRight / len(anglesRight)
    averageLeft = sumLeft / len(anglesLeft)

    print("Average Right: ", averageRight)
    print("Average Left: ", averageLeft)

    if ((averageRight > 160 and averageRight < 180 ) or (averageLeft > 160  and averageLeft < 180)):
        return True
        
    return False


def calculateAnglesPullUps(bodyPoints):
    anglesRight = []
    anglesLeft = []
    for points in bodyPoints:
        RShoulder = points["RShoulder"]
        RElbow = points["RElbow"]
        RWrist = points["RWrist"]
        LShoulder = points["LShoulder"]
        LElbow = points["LElbow"]
        LWrist = points["LWrist"]

        angleRight = calculateAngle(RShoulder, RElbow, RWrist)
        angleLeft = calculateAngle(LShoulder, LElbow, LWrist)

        anglesRight.append(angleRight)
        anglesLeft.append(angleLeft)


    plot_regression(anglesRight, 'regression_plot_right.png')
    plot_regression(anglesLeft, 'regression_plot_left.png')

    sumRight = sum(anglesRight) 
    sumLeft = sum(anglesLeft)

    averageRight = sumRight / len(anglesRight)
    averageLeft = sumLeft / len(anglesLeft)

    print("Average Right: ", averageRight)
    print("Average Left: ", averageLeft)

    if ((averageRight > 90 and averageRight < 97 ) and (averageLeft > 90  and averageLeft < 97)):
        return False
        
    return True

def calculateAngleShoulderPress(bodyPoints):
    angles = []
    for points in bodyPoints:
        RShoulder = points["RShoulder"]
        RElbow = points["RElbow"]
        RWrist = points["RWrist"]
        LShoulder = points["LShoulder"]
        LElbow = points["LElbow"]
        LWrist = points["LWrist"]

        angleRight = calculateAngle(RShoulder, RElbow, RWrist)
        angleLeft = calculateAngle(LShoulder, LElbow, LWrist)

        angles.append(angleRight)
        angles.append(angleLeft)

    plot_regression(angles, 'correcto.png')

    sumAngles = sum(angles) 
    average = sumAngles / len(angles)

    print("Average: ", average)

    if (average > 90 and average < 100):
        return True
        
    return False

