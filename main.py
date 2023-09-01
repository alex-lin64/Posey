import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pyfirmata

from threading import Thread
from threading import Timer
import time

from utils.processing import preprocess


# global vars
SQUAT_COUNT = 0
POSITION = 1  # 1 is up position, 0 is down
PROBABILITY = 0.00
PUNISH_CLOCK = 10
BOARD = None
SHUTDOWN = False  # shut down main loop within negative_reinforcement()


def update_count():
    """
    Monitors current squat position and updates the squat count accordingly.  
    
    Runs as a daemon thread
    """
    global SQUAT_COUNT
    global POSITION
    prev_pos = POSITION

    while True:
        if prev_pos and not POSITION:
            SQUAT_COUNT += 1
        prev_pos = POSITION
        time.sleep(0.1)

def punish():
    """
    Sends serial signal to arduino to open relay
    """
    BOARD.digital[7].write(1)
    time.sleep(1)
    BOARD.digital[7].write(0)

def negative_reinforcement():
    """
    Monitors squat count, will activate punshiment feature if squat count does 
    not increase every five seconds

    Runs as a daemon thread
    """
    global PUNISH_CLOCK
    global BOARD

    # only init arduino if it is being used
    print(f"Waiting for arduino connection...")
    while not BOARD:
        try:
            BOARD = pyfirmata.Arduino('COM3')
            print(f"Arduino connected!")
        except Exception as e:
            continue
        time.sleep(7)
    
    # timer class - done here as to make sure it is a daemon
    def newTimer():
        global t
        t = Timer(10.0, punish)
        return time.time()
    # init timer
    newTimer()
    
    # once board is connected serially, check for punishment
    prev_squat_count = SQUAT_COUNT
    while not SHUTDOWN:
        # timer
        if not t.is_alive():
            # gives time for previous punishment to execute
            time.sleep(1)
            start_time = newTimer()
            t.start()
        # check for changes in squat count
        elif SQUAT_COUNT > prev_squat_count:
            prev_squat_count = SQUAT_COUNT
            t.cancel()
            start_time = newTimer()
            t.start()
        # calculate time left before punishment
        time_left = 10 - int(time.time() - start_time)
        PUNISH_CLOCK = time_left if time_left >= 0 else 0


def main():
    """
    Main loop for Posey
    """
    global POSITION
    global PROBABILITY
    global SHUTDOWN

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

    # start squat count daemon
    print('Starting squat count task...')
    count_daemon = Thread(target=update_count, daemon=True, name='squat_count')
    count_daemon.start()

    # init punishment thread, only start when key 'p' (in main loop)
    punish_daemon = Thread(target=negative_reinforcement, daemon=True, name='punish')

    # start live stream 
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            _, frame = cap.read()
            
            # changing display colors
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False
            results = pose.process(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # fixes webcam opening late bug
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

                # prev_position = position
                POSITION = 1 if up >= down else 0
                PROBABILITY = str(round(up, 2)) if up >= down else str(round(down, 2))

            except Exception as e:
                print(f'Inference error: {e}')

            # display squat classification
            cv2.putText(
                img=frame,
                text=f"Count: {str(SQUAT_COUNT)}",
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
                text=("up" if POSITION else "down"),
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
                text=PROBABILITY,
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
                text=f"Squirt: {str(PUNISH_CLOCK)}",
                org=(10, 60),
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
            if k == ord('p'):
                # start punish daemon
                print('Starting punish task...')
                punish_daemon.start()
            elif k == ord('m'):
                print('Stopping punish task...')
                
            elif k == ord('q'):
                # quit by pressing 'q'
                break

    SHUTDOWN = True
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()