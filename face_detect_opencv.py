#!/usr/bin/python

import cv2, os, sys, platform
import numpy as np
from PIL import Image

class ActorRecognition():
    def __init__(self, training_path, test_path, **kwargs):
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
        self.recognizer = cv2.face.createLBPHFaceRecognizer()
        self.labels = None
        self.images = None
        self.images_path = training_path
        self.test_video_path = test_path
        self.image_label_dict = None

    def recognize_face(self, face_image):
        """ Predict the face
        
        Arguments:
            face_image {Numpy Array} -- Grayscaled image of just the face. Detect face before passing image to the function
        """
        predict_image = np.array(face_image, 'uint8')
        faces = faceCascade.detectMultiScale(predict_image)
        for (x, y, w, h) in faces:
            nbr_predicted = recognizer.predict(predict_image[y: y + h, x: x + w])
            print("Predicted : %d" % nbr_predicted)

    def map_images_to_labels(self):
        """Map numeric label for all the folders
        
        Returns:
            list -- Nested list contains folder name and the label  
        """
        folders = ActorRecognition.get_folder_contents(self.images_path)
        self.label_generator_for_folders(self.images_path)
        image_labels = []
        for folder in folders:
            files = ActorRecognition.get_folder_contents(folder)
            for image_file in files:
                label = ActorRecognition.get_folder_name_from_full_path(folder)
                image_labels.append([image_file,self.image_label_dict[label]])

        return image_labels
    
    def generate_labels(self):
        pass

    @staticmethod
    def get_folder_contents(path):
        """ Returns folder contents
        
        Arguments:
            path {string} -- Full path of the filder
        """
        return [os.path.join(path, f) for f in os.listdir(path) if not f.startswith('.')]
    
    @staticmethod
    def get_folder_name_from_full_path(full_path):
        """Get folder name from the full path
        
        Arguments:
            full_path {string} -- Full path of the folder
        
        Returns:
            [string] -- Folder name 
        """
        has_windows = True if platform.system() == 'Windows' else False
        if has_windows:
            delimiter = '\\'
        else:
            delimiter = '/'
        full_path = full_path[:-1] if full_path.endswith(delimiter) else full_path
        last_occurance = full_path.rfind(delimiter)
        return full_path[ last_occurance+1 : ]
    
    def label_generator_for_folders(self, path):
        """Generate label for the folders
        
        Arguments:
            path {string} -- Full path of the folder
        
        Returns:
            dict -- Dict containg the labels
        """
        folder_label_dict = dict()
        folders = ActorRecognition.get_folder_contents(path)
        for index, folder in enumerate(folders):
            folder_label_dict[ActorRecognition.get_folder_name_from_full_path(folder)] = index + 1
        
        self.image_label_dict = folder_label_dict
        return folder_label_dict

    def train_model(self, show_images = False):
        """Train model for the actor face recognition
        
        Keyword Arguments:
            show_images {boolean} -- Whether to show the images when training the model (default: {False})
        """
        print("Training Started")
        training_data = self.map_images_to_labels()
        images = []
        labels = []
        length = len(training_data)
        index = 0
        for image_path,label in training_data:
            try:
                image_pil = Image.open(image_path).convert('L')
            except Exception:
                pass

            image = np.array(image_pil, 'uint8')
            faces = self.face_cascade.detectMultiScale(image, minNeighbors = 15)
            for (x, y, w, h) in faces:
                images.append(image[y: y + h, x: x + w])
                labels.append(label)
                if show_images:
                    cv2.rectangle(image, (x,y), (x+w,y+h),(0,255,0),2)
        
            index = index + 1
            print("%d Complete..." % int(index*100/length) )
            if show_images:
                cv2.imshow("%s" % name[0], image)
                if cv2.waitKey(1) != -1:
                    cv2.destroyAllWindows()

        if show_images:
            cv2.destroyAllWindows()
        self.recognizer.train(images, np.array(labels))
        print("Training Finished")
    
    def play_video_in_cv2(self,video_path, highlight_faces = True):
        """Play test video and predict face
        
        Arguments:
            video_path {string} -- 
        
        Keyword Arguments:
            highlight_faces {boolean} -- [description] (default: {True})
        """
        print("Video Path : %s" % video_path)
        vc = cv2.VideoCapture(video_path)
        fps = vc.get(cv2.CAP_PROP_FPS)
        size = (int(vc.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        success, read = vc.read()
        print(success)
        i = 0
        while success:
            success, read = vc.read()
            i = i + 10.0
            vc.set(cv2.CAP_PROP_POS_FRAMES, i )
            gray_image = cv2.cvtColor(read, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray_image, minNeighbors = 15, minSize = (50,50))
            if highlight_faces:
                for (x,y,w,h) in faces:
                    cv2.rectangle(read, (x,y), (x+w,y+h),(0,255,0),2)
                    nbr_predicted = self.recognizer.predict(gray_image[y: y + h, x: x + w])
                    name = [ name for name in self.image_label_dict if self.image_label_dict[name] == nbr_predicted ]
                    print("Predicted : %s" % name[0])
                    cv2.putText(read, name[0], (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)

            cv2.imshow("Video", read)
            if cv2.waitKey(1) != -1:
                vc.release()
                cv2.destroyAllWindows()


if __name__ == '__main__':
    training_data_path =sys.argv[1]
    test_data_path = sys.argv[2]
    ob = ActorRecognition(training_data_path,test_data_path)
    ob.train_model(show_images = False)
    ob.play_video_in_cv2(test_data_path)
