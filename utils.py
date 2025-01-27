import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np


def validatePoses(bodyPoints, exercise):
    if (exercise == "ts1"):
        print("-----------------")
        print("ts1")
        print("-----------------")
        return validatePushUps(bodyPoints)
    elif (exercise == "ts2"):
        print("-----------------")
        print("ts2")
        print("-----------------")
        return validatePullUps(bodyPoints)
    elif (exercise == "ts3"):
        print("-----------------")
        print("ts3")
        print("-----------------")
        return validateShoulderPress(bodyPoints)
    elif exercise == "ts4":
        print("-----------------")
        print("ts4")
        print("-----------------")
        return validateRowing(bodyPoints)
    elif exercise == "ti1":
        print("-----------------")
        print("ti1")
        print("-----------------")
        return validateSquats(bodyPoints)
    elif exercise == "ti2":
        print("-----------------")
        print("ti2")
        print("-----------------")
        return validateDeadLift(bodyPoints)
    elif exercise == "ti3":
        print("-----------------")
        print("ti3")
        print("-----------------")
        return validateLunges(bodyPoints)
    elif exercise == "ti4":
        print("-----------------")
        print("ti4")
        print("-----------------")
        return validateBulgarianSquats(bodyPoints)

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
    repetitions = 0
    correctRepetitions = 0
    movement_phase = "initial"
    highest = 0
    lowest = 0
    tolerance = 40
    downPositionValid = False
    upPositionValid = False

    for points in bodyPoints:
        currentPosition = points['RHip'][1]

        RShoulder = points["RShoulder"]
        LShoulder = points["LShoulder"]
        RElbow = points["RElbow"]
        LElbow = points["LElbow"]
        RWrist = points["RWrist"]
        LWrist = points["LWrist"]

        rAngle = calculateAngle(RShoulder, RElbow, RWrist)
        lAngle = calculateAngle(LShoulder, LElbow, LWrist)

        if movement_phase == "initial":
            highest = currentPosition - tolerance
            lowest = currentPosition
            movement_phase = "down"
        
        if currentPosition < highest and movement_phase == "initial":
            movement_phase = "up"
        
        if currentPosition < lowest:
            lowest = currentPosition 
        
        if currentPosition > lowest + tolerance and movement_phase == "up":
            movement_phase = "down"
            highest = currentPosition
            repetitions += 1
            
            if rAngle > 250 and rAngle < 300 or lAngle > 250 and lAngle < 300:
                downPositionValid = True

            if downPositionValid and upPositionValid:
                correctRepetitions += 1
                downPositionValid = False
                upPositionValid = False
        
        if currentPosition > highest:
            highest = currentPosition

        if currentPosition < highest - tolerance and movement_phase == "down":
            movement_phase = "up"
            lowest = currentPosition

            if rAngle > 170 and rAngle < 205 or lAngle > 170 and lAngle < 205:
                upPositionValid = True

    message = correctRepetitions == 1 and "repetición" or "repeticiones"

    return {
        "is_Valid": correctRepetitions / repetitions >= 0.7,
        "message": f"Haz hecho {correctRepetitions} {message} correctamente."
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

    message = correctRepetitions == 1 and "repetición" or "repeticiones"

    return {
        "is_Valid": correctRepetitions / repetitions >= 0.7,
        "message": f"Haz hecho {correctRepetitions} {message} correctamente."
    }
    
def validateShoulderPress(bodyPoints):
    repetitions = 0
    is_down = False
    correctRepetitions = 0
    downPositionValid = False
    upPositionValid = False

    for points in bodyPoints:
        RWrist = points["RWrist"]
        RElbow = points["RElbow"]
        LElbow = points["LElbow"]
        RShoulder = points["RShoulder"]
        LShoulder = points["LShoulder"]
        RHip = points["RHip"]
        LHip = points["LHip"]

        rAngle = calculateAngle(RElbow, RShoulder, RHip)
        lAngle = calculateAngle(LHip, LShoulder, LElbow)

        angle = calculateAngle(RShoulder, RElbow, RWrist)

        if angle < 95:
            is_down = True
            if rAngle > 103 and rAngle < 112 and lAngle > 103 and lAngle < 112:
                upPositionValid = True
        elif angle > 160 and angle < 190 and is_down:
            repetitions += 1
            is_down = False 
            if rAngle > 161 and rAngle < 178 and lAngle > 161 and lAngle < 178:
                downPositionValid = True

            if downPositionValid and upPositionValid:
                correctRepetitions += 1
                downPositionValid = False
                upPositionValid = False

    message = correctRepetitions == 1 and "repetición" or "repeticiones"

    return {
        "is_Valid": correctRepetitions / repetitions >= 0.7,
        "message": f"Haz hecho {correctRepetitions} {message} correctamente."
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


def validateSquats(bodyPoints):
    repetitions = 0
    correctRepetitions = 0
    movement_phase = "initial"
    highest = 0
    lowest = 0
    tolerance = 40
    downPositionValid = False
    upPositionValid = False

    for points in bodyPoints:
        currentPosition = points['RHip'][1]

        RHip = points["RHip"]
        LHip = points["LHip"]
        RKnee = points["RKnee"]
        LKnee = points["LKnee"]
        RAnkle = points["RAnkle"]
        LAnkle = points["LAnkle"]
        RShoulder = points["RShoulder"]
        LShoulder = points["LShoulder"]

        rAngle = calculateAngle(RHip, RKnee, RAnkle)
        lAngle = calculateAngle(LHip, LKnee, LAnkle)
        rAngleShoulder = calculateAngle(RKnee, RHip, RShoulder)
        lAngleShoulder = calculateAngle(LKnee, LHip, LShoulder)

        if movement_phase == "initial":
            highest = currentPosition - tolerance
            lowest = currentPosition
            movement_phase = "down"
        
        if currentPosition < highest and movement_phase == "initial":
            movement_phase = "up"
        
        if currentPosition < lowest:
            lowest = currentPosition 
        
        if currentPosition > lowest + tolerance and movement_phase == "up":
            movement_phase = "down"
            highest = currentPosition
            repetitions += 1
            
            if rAngle > 275 and rAngle < 292 or lAngle > 275 and lAngle < 292 and rAngleShoulder > 290 and rAngleShoulder < 302 or lAngleShoulder > 290 and lAngleShoulder < 302:
                downPositionValid = True

            if downPositionValid and upPositionValid:
                correctRepetitions += 1
                downPositionValid = False
                upPositionValid = False
            
        if currentPosition > highest:
            highest = currentPosition

        if currentPosition < highest - tolerance and movement_phase == "down":
            movement_phase = "up"
            lowest = currentPosition

            if (rAngle > 205 and rAngle < 219) or (lAngle > 205 and lAngle < 219) and (rAngleShoulder > 209 and rAngleShoulder < 221) or (lAngleShoulder > 209 and lAngleShoulder < 221):
                upPositionValid = True

    message = correctRepetitions == 1 and "repetición" or "repeticiones"

    return {
        "is_Valid": correctRepetitions / repetitions >= 0.7,
        "message": f"Haz hecho {correctRepetitions} {message} correctamente."
    }

def validateDeadLift(bodyPoints):
    repetitions = 0
    correctRepetitions = 0
    movement_phase = "initial"
    highest = 0
    lowest = 0
    tolerance = 40
    downPositionValid = False
    upPositionValid = False

    for points in bodyPoints:
        currentPosition = points['RWrist'][1]

        RHip = points["RHip"]
        LHip = points["LHip"]
        RKnee = points["RKnee"]
        LKnee = points["LKnee"]
        RAnkle = points["RAnkle"]
        LAnkle = points["LAnkle"]
        RShoulder = points["RShoulder"]
        LShoulder = points["LShoulder"]

        rAngle = calculateAngle(RHip, RKnee, RAnkle)
        lAngle = calculateAngle(LHip, LKnee, LAnkle)
        rAngleShoulder = calculateAngle(RKnee, RHip, RShoulder)
        lAngleShoulder = calculateAngle(LKnee, LHip, LShoulder)

        if movement_phase == "initial":
            highest = currentPosition - tolerance
            lowest = currentPosition
            movement_phase = "down"
        
        if currentPosition < highest and movement_phase == "initial":
            movement_phase = "up"
        
        if currentPosition < lowest:
            lowest = currentPosition 
        
        if currentPosition > lowest + tolerance and movement_phase == "up":
            movement_phase = "down"
            highest = currentPosition
            repetitions += 1
            
            if rAngle > 235 and rAngle < 252 or lAngle > 235 and lAngle < 252 and rAngleShoulder > 263 and rAngleShoulder < 278 or lAngleShoulder > 263 and lAngleShoulder < 278:
                downPositionValid = True

            if downPositionValid and upPositionValid:
                correctRepetitions += 1
                downPositionValid = False
                upPositionValid = False
        
        if currentPosition > highest:
            highest = currentPosition

        if currentPosition < highest - tolerance and movement_phase == "down":
            movement_phase = "up"
            lowest = currentPosition

            if rAngle > 178 and rAngle < 219 or lAngle > 178 and lAngle < 219 and rAngleShoulder > 209 and rAngleShoulder < 221 or lAngleShoulder > 209 and lAngleShoulder < 221:
                upPositionValid = True

    message = correctRepetitions == 1 and "repetición" or "repeticiones"

    return {
        "is_Valid": correctRepetitions / repetitions >= 0.7,
        "message": f"Haz hecho {correctRepetitions} {message} correctamente."
    }

def validateLunges(bodyPoints):
    repetitions = 0
    correctRepetitions = 0
    movement_phase = "initial"
    highest = 0
    lowest = 0
    tolerance = 30
    downPositionValid = False
    upPositionValid = False

    for points in bodyPoints:
        currentPosition = points['LHip'][1]

        RHip = points["RHip"]
        LHip = points["LHip"]
        RKnee = points["RKnee"]
        LKnee = points["LKnee"]
        RAnkle = points["RAnkle"]
        LAnkle = points["LAnkle"]
        RShoulder = points["RShoulder"]
        LShoulder = points["LShoulder"]

        rAngle = calculateAngle(RHip, RKnee, RAnkle)
        lAngle = calculateAngle(LHip, LKnee, LAnkle)
        rAngleShoulder = calculateAngle(RKnee, RHip, RShoulder)
        lAngleShoulder = calculateAngle(LKnee, LHip, LShoulder)

        if movement_phase == "initial":
            highest = currentPosition - tolerance
            lowest = currentPosition
            movement_phase = "down"
        
        if currentPosition < highest and movement_phase == "initial":
            movement_phase = "up"
        
        if currentPosition < lowest:
            lowest = currentPosition 
        
        if currentPosition > lowest + tolerance and movement_phase == "up":
            movement_phase = "down"
            highest = currentPosition
            repetitions += 1
            
            if rAngle > 223 and rAngle < 242 and lAngle > 265 and lAngle < 279 and rAngleShoulder > 263 and rAngleShoulder < 278 or lAngleShoulder > 263 and lAngleShoulder < 278:
                downPositionValid = True

            if downPositionValid and upPositionValid:
                correctRepetitions += 1
                downPositionValid = False
                upPositionValid = False
        
        if currentPosition > highest:
            highest = currentPosition

        if currentPosition < highest - tolerance and movement_phase == "down":
            movement_phase = "up"
            lowest = currentPosition

            if rAngle > 197 and rAngle < 206 and lAngle > 219 and lAngle < 231 and rAngleShoulder > 233 and rAngleShoulder < 247 or lAngleShoulder > 233 and lAngleShoulder < 247:
                upPositionValid = True

    message = correctRepetitions == 1 and "repetición" or "repeticiones"

    return {
        "is_Valid": correctRepetitions / repetitions >= 0.7,
        "message": f"Haz hecho {correctRepetitions} {message} correctamente."
    }

def validateBulgarianSquats(bodyPoints):
    repetitions = 0
    correctRepetitions = 0
    movement_phase = "initial"
    highest = 0
    lowest = 0
    tolerance = 30
    downPositionValid = False
    upPositionValid = False

    for points in bodyPoints:
        currentPosition = points['LHip'][1]

        RHip = points["RHip"]
        LHip = points["LHip"]
        RKnee = points["RKnee"]
        LKnee = points["LKnee"]
        RAnkle = points["RAnkle"]
        LAnkle = points["LAnkle"]
        RShoulder = points["RShoulder"]
        LShoulder = points["LShoulder"]

        rAngle = calculateAngle(RHip, RKnee, RAnkle)
        lAngle = calculateAngle(LHip, LKnee, LAnkle)
        rAngleShoulder = calculateAngle(RKnee, RHip, RShoulder)
        lAngleShoulder = calculateAngle(LKnee, LHip, LShoulder)

        if movement_phase == "initial":
            highest = currentPosition - tolerance
            lowest = currentPosition
            movement_phase = "down"
        
        if currentPosition < highest and movement_phase == "initial":
            movement_phase = "up"
        
        if currentPosition < lowest:
            lowest = currentPosition 
        
        if currentPosition > lowest + tolerance and movement_phase == "up":
            movement_phase = "down"
            highest = currentPosition
            repetitions += 1
            
            if rAngle > 223 and rAngle < 242 and lAngle > 265 and lAngle < 279 and rAngleShoulder > 263 and rAngleShoulder < 278 or lAngleShoulder > 263 and lAngleShoulder < 278:
                downPositionValid = True

            if downPositionValid and upPositionValid:
                correctRepetitions += 1
                downPositionValid = False
                upPositionValid = False
            
            print("Rangle: ", rAngle)
            print("Langle: ", lAngle)
            print("RangleShoulder: ", rAngleShoulder)
            print("LangleShoulder: ", lAngleShoulder)
            print("Repetitions: ", repetitions)
            print("Correct Repetitions: ", correctRepetitions)
            print("Movement Phase: ", movement_phase)
            print("---------------------------------------------")
        
        if currentPosition > highest:
            highest = currentPosition

        if currentPosition < highest - tolerance and movement_phase == "down":
            movement_phase = "up"
            lowest = currentPosition

            if rAngle > 197 and rAngle < 206 and lAngle > 219 and lAngle < 231 and rAngleShoulder > 233 and rAngleShoulder < 247 or lAngleShoulder > 233 and lAngleShoulder < 247:
                upPositionValid = True

            print("Rangle: ", rAngle)
            print("Langle: ", lAngle)
            print("RangleShoulder: ", rAngleShoulder)
            print("LangleShoulder: ", lAngleShoulder)
            print("Repetitions: ", repetitions)
            print("Correct Repetitions: ", correctRepetitions)
            print("Movement Phase: ", movement_phase)
            print("---------------------------------------------")

    message = correctRepetitions == 1 and "repetición" or "repeticiones"

    return {
        "is_Valid": correctRepetitions / repetitions >= 0.7,
        "message": f"Haz hecho {correctRepetitions} {message} correctamente."
    }