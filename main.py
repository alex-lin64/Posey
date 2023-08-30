import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

from utils.processing import preprocess


def main():
    # init mp pose objects
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    # load TFLite model, allocate tensors
    interpreter = tf.lite.Interpreter(model_path='model\\tf_lite_model\\squat_classifier.tflite')
    interpreter.allocate_tensors()
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # start video capture
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # start live stream 
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            
            # changing display colors
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False
            results = pose.process(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # if webcam opens late, results will be None
            if not results.pose_landmarks:
                cv2.imshow("Webcam Feed", frame)
                # quit by pressing 'q'
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                continue

            # squat classifier - set input data
            input_data = preprocess(results.pose_landmarks.landmark)

            try:
                interpreter.set_tensor(input_details[0]['index'], input_data)
                # squat classifier - invoke inference
                interpreter.invoke()

                # squat classifier - invoke inference
                # index 0 is down position, index 1 is up position
                output_data = interpreter.get_tensor(output_details[0]['index'])

                # interpret results
                position = "up" if output_data[0][1] >= output_data[0][0] else "down"

                # display squat classification
                cv2.putText(
                    img=frame,
                    text=position,
                    org=(10, frame.shape[0] - 10),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=1,
                    color=(255,255,255),
                    thickness=1,
                    lineType=cv2.LINE_AA
                )
            except Exception as e:
                print(e)

            # visualize pose
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                )

            cv2.imshow("Webcam Feed", frame)

            # quit by pressing 'q'
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()