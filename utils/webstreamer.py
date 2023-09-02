import cv2 
from threading import Thread 


class WebcamStream:
    
    def __init__(self, src=0):
        """
        Webstreamer, captures frames from src with cv2, runs as daemon thread
        """
        self.src = src # default 0 for main camera 
        
        # opening video capture stream 
        self.cap = cv2.VideoCapture(self.src)

        # health check
        if self.cap.isOpened() is False :
            print("Error accessing webcam stream, exiting.")
            exit(0)
        
        # check fps
        fps_input_stream = int(self.cap.get(5)) # hardware fps
        print("FPS of input stream: {}".format(fps_input_stream))
            
        # reading a single frame from vcap stream for initializing 
        self.grabbed, self.frame = self.cap.read()

        if not self.grabbed:
            print('No more frames to read, exiting.')
            exit(0)

        # self.stopped is initialized to False 
        self.stopped = True

        # thread instantiation  
        self.t = Thread(target=self.update, daemon=True, args=())
        
    def start(self):
        """
        Starts the daemon thread
        """
        self.stopped = False
        self.t.start()

    # method passed to thread to read next available frame  
    def update(self):
        """
        Read the next available frame
        """
        while True:
            # break if quit from main
            if self.stopped:
                break
            # grab next frame
            self.grabbed, self.frame = self.cap.read()
            if not self.grabbed:
                print('No more frames to read, exiting.')
                self.stopped = True
                break 
        self.cap.release()

    def read(self):
        """
        Returns the next frame
        """
        return self.frame

    def stop(self):
        """
        Stops the thread
        """
        self.stopped = True