import time
import threading
from queue import Queue


class SquatCounter(threading.Thread):

    def __init__(self):
        """
        Monitors current squat position and updates the squat count accordingly.  
    
        *Runs as a daemon thread*
        """
        super().__init__()
        self.daemon = True
        self.squat_count = 0
        self.position = Queue()

    def run(self):
        """
        Monitors current squat position and updates the squat count accordingly.  
        
        Runs as a daemon thread
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
    

