import time
import threading
from queue import Queue


class SquatCounter(threading.Thread):
    """
    Monitors current squat position and updates the squat count accordingly.  

    *Runs as a daemon thread*
    """
    def __init__(self):
        """
        Constructor
        """
        super().__init__()
        self.daemon = True
        self.squat_count = 0
        self.position = Queue()  # stores squat positions as main loop infers

    def run(self):
        """
        Monitors current squat position and updates the squat count accordingly.  
        """
        prev_pos = 1
        cur_pos = 1
        while True:
            # if queue if empty, continue
            if self.position.empty():
                time.sleep(0.2)
                continue
            # get cur position
            cur_pos = self.position.get()
            if prev_pos and not cur_pos:
                self.squat_count += 1
            prev_pos = cur_pos
    

