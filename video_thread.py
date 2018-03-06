import threading
import time
import cv2
import signal, sys
import random

  

class VideoCaptureThread(threading.Thread):
    exit_flag = False
    def __init__(self, video_path, frame_skip = 10, size = 480):
        threading.Thread.__init__(self)
        self._video = video_path
        self._read_queue = []
        self.vc = cv2.VideoCapture(video_path)
        self._exit = False
        self._read_started = False
        self._last_read = None
        self._buffer_size = 5000
        self._frame_skip = frame_skip
        self._height = 1
        self._width = 1
        self._size = size
        signal.signal(signal.SIGINT, VideoCaptureThread.end_process)
        signal.signal(signal.SIGTERM, VideoCaptureThread.end_process)  
    
    @staticmethod
    def end_process(a, b):
        VideoCaptureThread.exit_flag = True
        print("I quit...")
    
    def run(self):
      print("Starting Video")
      self.read_video()
      print("Video Ended")

    def read_video(self):
        vc = self.vc
        fps = vc.get(cv2.CAP_PROP_FPS)
        size = (int(vc.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if self._size == 0:
            self._width, self._height = size
        else:
            video_ratio = size[0]/size[1]
            self._width = self._size
            self._height = int(self._size / video_ratio)

        success, read = vc.read()
        self._last_read = success, read
        frame = 0
        while success:
            if VideoCaptureThread.exit_flag:
                return
            while len(self._read_queue) == self._buffer_size:
                pass
            frame = frame +  random.randint(1, self._frame_skip) #self._frame_skip
            vc.set(cv2.CAP_PROP_POS_FRAMES, frame )
            success, read = vc.read()
            self._last_read = success, read
            self._read_started = True
            try:
                res = cv2.resize(read,( self._width, self._height ))
                self._read_queue.append([success, res])
            except Exception:
                self._read_queue.append([success, read])
            
            
        VideoCaptureThread.exit_flag = True
    
    def play_video(self):
        while len(self._read_queue) == 0:
            pass
        success, read = self._read_queue.pop(0)
        if VideoCaptureThread.exit_flag:
            return (False, 0)
        return success, read


