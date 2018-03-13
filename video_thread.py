import threading
import time
import cv2
import signal, sys
import random

  

class VideoCaptureThread(threading.Thread):
    exit_flag = False
    def __init__(self, video_path, frame_skip = 10, size = None):
        threading.Thread.__init__(self)
        VideoCaptureThread.exit_flag = False
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
        print("Thread End")
    
    def run(self):
      self.read_video()

    def release_resource(self):
        VideoCaptureThread.exit_flag = True
    
    def read_video(self):
        vc = self.vc
        fps = vc.get(cv2.CAP_PROP_FPS)
        size = (int(vc.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if self._size == None or self._size >= size[0]:
            self._width, self._height = size
            self._size = None
        else:
            video_ratio = size[0]/size[1]
            self._width = self._size
            self._height = int(self._size / video_ratio)

        success, read = vc.read()
        self._last_read = success, read
        frame = 0
        while success:
            if VideoCaptureThread.exit_flag:
                vc.release()
                print("INFO : Exit Flag set to true. Releasing file")
                return
            while len(self._read_queue) == self._buffer_size:
                pass
            frame = frame +  random.randint(1, self._frame_skip) #self._frame_skip
            
            vc.set(cv2.CAP_PROP_POS_FRAMES, frame )
            success, read = vc.read()
            #print("Width : %s\t\tHeight : %s" % (str(self._width), str(self._height)) )
            #print("Parameter : %s" % str(self._size))
                
            try:
                if self._size is not None:
                    res = cv2.resize(read, (self._width, self._height))
                    self._read_queue.append([success, res, frame, 0])
                    self._last_read = success, res
                else:
                    self._read_queue.append([success, read, frame, 0])
                    self._last_read = success, read
            except Exception:
                continue

            self._read_started = True
            
            
        VideoCaptureThread.exit_flag = True
        vc.release()
    
    def play_video(self):
        while len(self._read_queue) == 0:
            pass
        success, read, frame, buffer = self._read_queue.pop(0)
        if VideoCaptureThread.exit_flag:
            return (False, 0, 0, 0)
        return success, read, frame, buffer


