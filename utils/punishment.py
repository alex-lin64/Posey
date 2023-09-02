import time
import threading
import pyfirmata
from threading import Timer


def punish(board):
    """
    Sends serial signal to arduino to open relay

    :params:
        - board: pyfirmata object, represents arduino board
    """
    board.digital[7].write(1)
    time.sleep(1)
    board.digital[7].write(0)



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
        exit(1)

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