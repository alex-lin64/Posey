import time
from threading import Timer, Thread
import pyfirmata


class NegativeReinforcement():
    """
    Monitors squat count to ensure squats are performed in orderly manner, 
    or else...
    """
    def __init__(self, rest_time, event):
        """
        Constructor

        :params:
            - rest_time: int, the amount of time to do a squat before the 
                punishment is activated
            - event: threading.Event, signals for thread to exit
        """
        self.rest_time = rest_time
        self.time_left = 10
        self.count = 0

        self._start_time = None
        self._board = None
        self._paused = True  # punishment thread starts out paused
        self._timer = None

        self._punish_thread = Thread(
            target=self.negative_reinforcement, 
            daemon=True, 
            args=(event,)
            )

        # init timer, not started it since punish thread starts out paused
        self.new_timer()

    def _init_board(self):
        """
        Connects arduino board to the thread

        :returns:
            - True, if board successfully connects, False if board is already 
                connected or an exception is thrown
        """
        print(f"Waiting for arduino connection...")
        try:
            self._board = pyfirmata.Arduino('COM3')
            print(f"Arduino connected!")
            return True
        except Exception as e:
            print("Arduino board not found...try again")
            return False

    def start(self):
        """
        Starts the punish thread but in paused state
        """
        try:
            self._punish_thread.start()
        except Exception as e:
            print(f"punish thread error: {e}")

    def unpause(self):
        """
        Unpauses the punish thread
        """
        # if punishment already is running
        if not self._paused:
            print("Punishment is already running.")
            return
        # init board if board is None
        if not self._board and not self._init_board():
            return
        # unpause
        self._paused = False

    def pause(self):
        """
        Pauses the punish thread
        """
        # Can pause multiple times
        self._paused = True

    def new_timer(self):
        """
        Creates a new timer for each time punishment clock is reset

        **depends on garbage collection to get rid of old timers
        """
        self._timer = Timer(self.rest_time, self.punish, args=(self._board,))
        self._start_time = time.time()

    def punish(self, board):
        """
        Sends serial signal to arduino to open relay

        :params:
            - board: pyfirmata object, represents arduino board
        """
        # if board is not connected
        if not self._board:
            print("Error: no board detected.  Punish thread will be paused.  Please connect arduino first and then restart punishment (press key 'u').")
            self._paused = True
            return
        try:
            # if board is connected
            board.digital[7].write(1)
            time.sleep(1)
            board.digital[7].write(0)
        except Exception as e:
            print(f"Arduino error: {e}")
            print("Pausing punish thread.  Please reconnect arduino and then restart punishment with key 'u'")
            self._paused = True
            self._board = None

    def negative_reinforcement(self, event):
        """
        Monitors squat count, will activate punshiment feature if squat count does 
        not increase every five seconds

        :params:
            - event: threading.Event, signals for thread to exit
        """
        prev_count = 0

        while not event.isSet():
            # paused state
            if self._paused:
                if self._timer.is_alive():
                    self._timer.cancel()
                time.sleep(0.5)
                continue
            # timer thread dead means timer finished or hasn't started
            if not self._timer.is_alive():
                # gives time for previous punishment to execute
                time.sleep(0.5)
                self.new_timer()
                self._timer.start()
            elif self.count > prev_count:
                prev_count = self.count
                self._timer.cancel()
                self.new_timer()
                self._timer.start()
            # calculate time left before punishment
            time_left = 10 - int(time.time() - self._start_time)
            self.time_left = time_left if time_left >= 0 else 0
            time.sleep(0.5)

        # clean up timer thread
        if self._timer.is_alive():
            self._timer.cancel()