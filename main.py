import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pyfirmata

from threading import Thread, Timer, Event
import time

from utils.processing import preprocess
from utils.punishment import punish
from utils.squat_counter import SquatCounter
from utils.webstreamer import WebcamStream


# global vars
SQUAT_COUNT = 0
PUNISH_CLOCK = 10


def negative_reinforcement(event):
    """
    Monitors squat count, will activate punshiment feature if squat count does 
    not increase every five seconds

    :params:
        - event: 
    """
    global PUNISH_CLOCK

    # only init arduino if it is being used
    print(f"Waiting for arduino connection...")
    try:
        board = pyfirmata.Arduino('COM3')
        print(f"Arduino connected!")
    except Exception as e:
        print("Arduino board not found...try again")
        exit(0)

    # timer class - done here as to make sure global is in scope
    def newTimer():
        global t
        t = Timer(10.0, punish, args=(board,))
        return time.time()
    # init timer
    newTimer()
    
    # once board is connected serially, check for punishment
    prev_squat_count = SQUAT_COUNT
    while not event.isSet():
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
        time.sleep(0.5)
    # clean up timer thread
    t.cancel()

def main():
    """
    Main loop for Posey
    """

    # init displayed vars
    probabilty = 0.0
    position = 1

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

    # init punishment thread, start with key 'p', kills thread with 'm'\
    event = Event()
    punish_task = Thread(target=negative_reinforcement, args=(event,), name='punish')

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
                try:
                    if punish_task.is_alive():
                        print("Task is already started, end the current task by pressing m")
                        continue
                    event.clear()
                    print('Starting punish task...')
                    punish_task.start()
                except Exception as e:
                    print(e)
            elif k == ord('m'):
                print('Stopping punish task...')
                event.set()
                punish_task.join()
            elif k == ord('q'):
                # quit
                break 
    

    stream.stop()
    cv2.destroyAllWindows()

    event.set() 
    time.sleep(0.1)  # gives time for negative_reinforcement to shutdown


if __name__ == '__main__':
    main()