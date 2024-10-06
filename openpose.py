import cv2 as cv
import numpy as np
from datetime import datetime
import os

def openPose(video):
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
    valid = True if video.endswith('1.mp4') else False
    video = video.split('/')[-1].split('.')[0]
    fps = cap.get(cv.CAP_PROP_FPS)
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    new_video = f'{timestamp}{video}.avi'
    video_writer = cv.VideoWriter(new_video, fourcc, fps, (frame_width, frame_height))

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
        "is_valid": valid
    }
