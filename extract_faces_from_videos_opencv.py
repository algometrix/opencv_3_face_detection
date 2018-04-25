#!/usr/bin/python

import cv2, os, sys, platform
import numpy as np
from PIL import Image
import argparse
#import tqdm

extracted_face = {}

class FaceExtractor():
    def __init__(self, extract_location, **kwargs):
        """Constructor for ActorRecognition
        
        Arguments:
            training_path {string}      --      Path of the training data. 
                                                Should contain folders with name 
                                                of the actor containing images of the actor

            test_path {string}          --      Full path of the test video path
            **kwargs {named arguments}  --      Named arguments for later use. NOT USED YET
        """
        self.cascade_path = "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(self.cascade_path)
        self.extract_location = extract_location

    def extract_faces_cv2(self,video_path, frame_skip, show_video, save_images, highlight_faces = True, extraction_folder =''):
        """Play test video and predict face
        
        Arguments:
            video_path {string} -- 
        
        Keyword Arguments:
            highlight_faces {boolean} -- [description] (default: {True})
        """
        #print("Video Path : %s" % video_path)
        location , file_name = os.path.split(video_path)
        vc   = cv2.VideoCapture(video_path)
        fps  = vc.get(cv2.CAP_PROP_FPS)
        size = (int(vc.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        resize_value = (1280, 720)
        original_frame_skip = frame_skip
        face_detected = False
        resize = False
        width, height = size
        success, read = vc.read()
        display_size = (640,480)
        for_show = cv2.resize(read, display_size)
        #print("Video Resolution : %s" % str(size))
        if False and width > 1280:
            print("Resolution larger than %s. Resizing video..." % str(resize_value))
            resize = True
        i = 0
        count = 0
        total_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
        mid_frame = total_frames
        step_1 = total_frames * 0.40
        step_2 = total_frames * 0.70
        while success:
            if resize and False:
                read = cv2.resize(read, resize_value)
            
            i = i + frame_skip
            vc.set(cv2.CAP_PROP_POS_FRAMES, i )
            gray_image = cv2.cvtColor(read, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray_image, minNeighbors = 12, minSize = (100,100))
            
            for (x,y,w,h) in faces:
                face_detected = True
                face_image = read[y:y+w, x:x+h]
                extracted_face[extraction_folder] = extracted_face[extraction_folder] + 1
                if save_images:
                    image_name = os.path.join(self.extract_location, extraction_folder, "%s_frame%d.jpg" % (file_name, count))
                    folder_path, dummy = os.path.split(image_name)
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)
                    cv2.imwrite(image_name, face_image)
                
                if highlight_faces:
                    cv2.rectangle(read, (x,y), (x+w,y+h),(0,255,0),2)

                count = count + 1

            
            if show_video: # and face_detected:
                for_show = cv2.resize(read, display_size)
                cv2.putText(for_show, "{} : {}".format(extraction_folder,extracted_face[extraction_folder]), (0,30), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
                cv2.imshow("Video", for_show)

            if len(faces):
                frame_skip = 2
            else:
                if i > step_1 and i < step_2:
                    frame_skip = 2 * original_frame_skip
                else:
                    frame_skip = original_frame_skip
                face_detected = False


            #print("=================== FRAME : %07d ===================" % frame_skip)

            if cv2.waitKey(1) != -1:
                vc.release()
                cv2.destroyAllWindows()

            success, read = vc.read()


def get_folder_contents(path):
    return [os.path.join(path, f) for f in os.listdir(path) if not f.startswith('.')]
    
'''
python extract_faces_from_videos_opencv.py --dest_folder "E:\Downloads\Training Stuff\Test Data" --frame_skip 50 --show_video true --path 
'''

if __name__ == '__main__':
    print("checking...")
    parser = argparse.ArgumentParser(description = 'Arguments for Face Recognition')
    parser.add_argument('--path', action = 'store', dest = 'video_path', required = True)
    parser.add_argument('--dest_folder', action = 'store', dest = 'dest_folder', required = True)
    parser.add_argument('--frame_skip',  action = 'store', dest = 'frame_skip',  default = 10.0,  type = float, required = False)
    parser.add_argument('--show_video',  action = 'store', dest = 'show_video',  default = False, type = bool,  required = False)
    parser.add_argument('--save_images', action = 'store', dest = 'save_images', default = True,  type = bool,  required = False)
    
    face_args   = parser.parse_args()
    video_path  = face_args.video_path
    dest_folder = face_args.dest_folder
    frame_skip  = face_args.frame_skip
    show_video  = face_args.show_video
    save_images = face_args.save_images
    ob          = FaceExtractor(dest_folder)

    dest_folder_contents = os.listdir(dest_folder)

    def traverse_folder(path, video_limit = 1000, face_limit = 2000):
        files = get_folder_contents(path)
        print("Number of files : {}".format(len(files)))
        for file_path in files:
            print("Extracting from {}".format(file_path))
            if not os.path.isdir(file_path):
                if len(files) > video_limit:
                    print("================ Limit Exceeded ================")
                    return
                folder_path, filename    = os.path.split(file_path)
                path       , folder_name = os.path.split(folder_path)
                if folder_name in dest_folder_contents:
                    print("{} folder exists\n".format(folder_name))
                    return
                try :
                    extracted_face[folder_name] = extracted_face[folder_name] + 0
                except Exception:
                    print("Exception")
                    extracted_face[folder_name] = 0
                
                if extracted_face[folder_name] > face_limit:
                    print("100 Faces extracted")
                    continue
                ob.extract_faces_cv2(file_path, frame_skip, show_video, save_images, extraction_folder = folder_name)
            else:
                print("{} is a folder".format(file_path))
                traverse_folder(file_path, video_limit = 2000)
    
    if os.path.isdir(video_path):
        print("{} is the video path".format(video_path))
        traverse_folder(video_path, video_limit = 2000)
    else:
        extracted_face[''] = 0
        ob.extract_faces_cv2(video_path, frame_skip, show_video, save_images)
    