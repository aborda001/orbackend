import os
from datetime import datetime
import cv2 as cv
import mediapipe as mp

def mediaPipeModel(video_path):
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    cap = cv.VideoCapture(video_path)

    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv.CAP_PROP_FPS)
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    video_path = video_path.split('/')[-1].split('.')[0]
    new_video = f'{timestamp}{video_path}.avi'
    video_writer = cv.VideoWriter(new_video, fourcc, fps, (frame_width, frame_height))
    body_points = []


    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                RWrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
                RElbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
                RShoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                LShoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                LElbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
                LWrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
                RHip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
                RKnee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
                RAnkle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
                LHip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
                LKnee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
                LAnkle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
                Nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]

                points = {
                    "RWrist": [RWrist.x, RWrist.y],
                    "RElbow": [RElbow.x, RElbow.y],
                    "RShoulder": [RShoulder.x, RShoulder.y],
                    "LShoulder": [LShoulder.x, LShoulder.y],
                    "LElbow": [LElbow.x, LElbow.y],
                    "LWrist": [LWrist.x, LWrist.y],
                    "RHip": [RHip.x, RHip.y],
                    "RKnee": [RKnee.x, RKnee.y],
                    "RAnkle": [RAnkle.x, RAnkle.y],
                    "LHip": [LHip.x, LHip.y],
                    "LKnee": [LKnee.x, LKnee.y],
                    "LAnkle": [LAnkle.x, LAnkle.y],
                    "Nose": [Nose.x, Nose.y]
                }

                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2),
                )

                for label, coord in points.items():
                    x = int(coord[0] * frame.shape[1])
                    y = int(coord[1] * frame.shape[0])

                    points[label] = [x, frame.shape[0] - y ]

                    # cv.putText(frame, label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
                body_points.append(points)

            video_writer.write(frame)

            # cv.imshow('Pose Detection', frame)

            if cv.waitKey(10) & 0xFF == ord('q'):
                break

    convert = f"ffmpeg -i {new_video} -vcodec libx264 {timestamp}{video_path}.mp4"
    os.system(convert)
    new_video = f"{timestamp}{video_path}.mp4"

    cap.release()
    cv.destroyAllWindows()
    video_writer.release()

    return {
        "video": new_video,
        "body_points":body_points
    }

def openPoseModel(video):
    BODY_PARTS = {
        "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
        "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
        "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Background": 14
    }

    POSE_PAIRS = [
        ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
        ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
        ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"],
        ["Neck", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"],
        ["Neck", "Nose"]
    ]

    image_width = 600
    image_height = 600

    net = cv.dnn.readNetFromTensorflow("graph_opt.pb")

    threshold = 0.1
    cap = cv.VideoCapture(video)
    video = video.split('/')[-1].split('.')[0]
    fps = cap.get(cv.CAP_PROP_FPS)
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    new_video = f'{timestamp}{video}.avi'
    video_writer = cv.VideoWriter(new_video, fourcc, fps, (frame_width, frame_height))
    body_points = []

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break 

        photo_height = img.shape[0]
        photo_width = img.shape[1]
        net.setInput(cv.dnn.blobFromImage(img, 1.0, (image_width, image_height), (127.5, 127.5, 127.5), swapRB=True, crop=False))

        output = net.forward()
        output = output[:, :15, :, :]

        points = []
        for i in range(len(BODY_PARTS)):
            heatMap = output[0, i, :, :]
            _, conf, _, point = cv.minMaxLoc(heatMap)
            x = (photo_width * point[0]) / output.shape[3]
            y = (photo_height * point[1]) / output.shape[2]
            points.append((int(x), int(y)) if conf > threshold else None)

        for pair in POSE_PAIRS:
            partFrom = pair[0]
            partTo = pair[1]
            assert(partFrom in BODY_PARTS)
            assert(partTo in BODY_PARTS)

            idFrom = BODY_PARTS[partFrom]
            idTo = BODY_PARTS[partTo]

            if points[idFrom] and points[idTo]:
                cv.line(img, points[idFrom], points[idTo], (0, 255, 0), 3)
                cv.ellipse(img, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
                cv.ellipse(img, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
        

        RWrist = points[4]
        RElbow = points[3]
        RShoulder = points[2]
        LShoulder = points[5]
        LElbow = points[6]
        LWrist = points[7]
        RHip = points[8]
        RKnee = points[9]
        RAnkle = points[10]
        LHip = points[11]
        LKnee = points[12]
        LAnkle = points[13]

        body_points.append({
            "RWrist": RWrist,
            "RElbow": RElbow,
            "RShoulder": RShoulder,
            "LShoulder": LShoulder,
            "LElbow": LElbow,
            "LWrist": LWrist,
            "RHip": RHip,
            "RKnee": RKnee,
            "RAnkle": RAnkle,
            "LHip": LHip,
            "LKnee": LKnee,
            "LAnkle": LAnkle
        })
      
        video_writer.write(img)
        # cv.imshow("OpenPose Output", img)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    convert = f"ffmpeg -i {new_video} -vcodec libx264 {timestamp}{video}.mp4"
    os.system(convert)
    new_video = f"{timestamp}{video}.mp4"

    cap.release()
    video_writer.release()
    cv.destroyAllWindows()

    return {
        "video": new_video,
        "body_points": body_points
    }


if __name__ == '__main__':
    from utils import validatePoses
    video = '2.mp4'

    if (input("Model: ") == '0'):
        result = mediaPipeModel(video)
    else:
        result = openPoseModel(video)
    
    isValid = validatePoses(result['body_points'], "ts2")

    print({
            'message': 'File uploaded successfully',
            'newVideo': result['video'],
            'isVideoValid': isValid,
        })
