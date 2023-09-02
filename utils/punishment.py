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
            - evebt: threading.Event, signals for thread to exit
        
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
            args=(event)
            )

        # init timer, not started it since punish thread starts out paused
        self.new_timer()

    def init_board(self):
        """
        Connects arduino board to the thread
        """
        # if boarded already connected or punishment already runnning 
        if self._board:
            print("Board already connected, please stop punishment and restart.")
            return False

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
        self._punish_thread.start()

    def unpause(self):
        """
        Unpauses the punish thread
        """
        if not self._paused:
            print("Punishment is already running.")
            return
        
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
            print("Error: please connect arduino before starting the punishment thread.")
            return
        # if board is connected
        board.digital[7].write(1)
        time.sleep(1)
        board.digital[7].write(0)

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
                time.sleep(0.5)
                continue

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