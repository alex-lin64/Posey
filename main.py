import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

from threading import Event
import time

from utils.processing import preprocess
from utils.squat_counter import SquatCounter
from utils.webstreamer import WebcamStream
from utils.punishment import NegativeReinforcement


def main():
    """
    Main loop for Posey
    """
    # init internal vars
    probabilty = 0.0
    position = 1
    rest_time = 10.0

    # init mp pose objects
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # load TFLite model, allocate tensors
    interpreter = tf.lite.Interpreter(model_path='model\\tf_lite_model\\squat_classifier.tflite')
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    try:
        # start video capture
        print('Starting squat count task...')
        stream = WebcamStream(src=0)
        stream.start()
    except Exception as e:
        print(f"WebcamStream error: {e}")
        exit(0)

    try:
        # start squat count daemon
        print('Starting squat count task...')
        count_daemon = SquatCounter()
        count_daemon.start()
    except Exception as e:
        print(f"SquatCounter error: {e}")
        exit(0)

    try:
        # init punishment thread, unpause with key 'u', pause thread with 'p'\
        print("Initializing punishment thread")
        event = Event()
        punish_daemon = NegativeReinforcement(rest_time=rest_time, event=event)
        punish_daemon.start()
    except Exception as e:
        print(f"NegativeReinforcement error: {e}")
        exit(0)

    # start live stream 
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while not stream.stopped:
            frame = stream.read()
            
            # changing display colors for inference
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False
            results = pose.process(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # if stream opens late, won't throw an exception
            if not results.pose_landmarks:
                cv2.imshow("Webcam Feed", frame)
                # quit by pressing 'q'
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                continue

            # squat classifier - set input data
            input_data = preprocess(results.pose_landmarks.landmark)

            try:
                # set input data to model
                interpreter.set_tensor(input_details[0]['index'], input_data)

                # squat classifier - invoke inference
                interpreter.invoke()

                # squat classifier - invoke inference
                output_data = interpreter.get_tensor(output_details[0]['index'])
                down, up = output_data[0][0], output_data[0][1]
                
                # update to latest position and probability
                position = 1 if up >= down else 0
                probabilty = str(round(up, 2)) if up >= down else str(round(down, 2))

                # pass new position to squat counter thread
                count_daemon.position.put(position)
                punish_daemon.count = count_daemon.squat_count
            except Exception as e:
                print(f'Inference error: {e}')

            # display squat classification
            cv2.putText(
                img=frame,
                text=f"Count: {str(count_daemon.squat_count)}",
                org=(10, 30),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=1,
                color=(255, 143, 23),
                thickness=2,
                lineType=cv2.LINE_AA
            )
            # display squat classification
            cv2.putText(
                img=frame,
                text=("up" if position else "down"),
                org=(10, frame.shape[0] - 40),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=1,
                color=(255, 143, 23),
                thickness=2,
                lineType=cv2.LINE_AA
            )
            # display squat classification probability
            cv2.putText(
                img=frame,
                text=probabilty,
                org=(10, frame.shape[0] - 10),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=1,
                color=(255, 143, 23),
                thickness=2,
                lineType=cv2.LINE_AA
            )
            # display punish clock
            cv2.putText(
                img=frame,
                text=f"Squirt: {str(punish_daemon.time_left)}",
                org=(10, 60),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=1,
                color=(255, 143, 23),
                thickness=2,
                lineType=cv2.LINE_AA
            )
            # display punishment state
            state = "Paused" if punish_daemon._paused else "Active"
            cv2.putText(
                img=frame,
                text=f"Punishment: {state}",
                org=(10, 90),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=1,
                color=(255, 143, 23),
                thickness=2,
                lineType=cv2.LINE_AA
            )

            # visualize pose
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                )

            cv2.imshow("Webcam Feed", frame)

            k = cv2.waitKey(10)
            if k == ord('u'):
                print('Starting punish task...')
                punish_daemon.unpause()
            elif k == ord('p'):
                print('Pausing punish task...')
                punish_daemon.pause()
            elif k == ord('q'):
                # quit
                print("Cleaning up threads")
                event.set()  # signals punish thread to exit, put here to migitage punishment executing after exiting
                break 

    stream.stop()
    cv2.destroyAllWindows()
    time.sleep(0.1)  # gives time for negative_reinforcement to shutdown
    print("Sucessfully exited!")


if __name__ == '__main__':
    main()