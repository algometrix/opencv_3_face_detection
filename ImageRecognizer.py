
# coding: utf-8

# In[1]:


from math import sqrt
from sklearn import neighbors
import os
import pickle
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import face_recognition
from face_recognition import face_locations
from face_recognition.face_recognition_cli import image_files_in_folder
import cv2
import argparse
import time
import sys


# In[ ]:


class ImageRecog():
    def __init__(self, label_name = 'generic'):
        self.cascade_path = "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(self.cascade_path)
        self._training_dir = None
        self._extracton_folder = None
        self._image_folder = None
        self._label_name = label_name
    
    def get_folder_contents(self, folder_path):
        return os.listdir(folder_path)
    
    def train(self, training_dir, n_neighbors = None, knn_algo = 'ball_tree', 
            image_face_only = False, sample_size = 200):
        X = []
        y = []
        train_dir = self._training_dir
        
        files = self.get_folder_contents(training_dir)
        print("Number of files found : %d" % len(files))
        for image_file in files:
            image_file_full_path = os.path.join(training_dir, image_file)
            image = face_recognition.load_image_file(image_file_full_path)
            faces_bboxes = face_locations(image)
            if faces_bboxes:
                X.append(face_recognition.face_encodings(image, known_face_locations=faces_bboxes)[0])
                y.append(self._label_name)
        
        if n_neighbors is None:
            n_neighbors = int(round(sqrt(len(X))))
            print("Chose n_neighbors automatically as:", n_neighbors)

        knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')        
        knn_clf.fit(X, y)
        return knn_clf
    
    def predict(self, classifier, image_path, DIST_THRESH = .5):
        X_img = face_recognition.load_image_file(image_path)
        X_faces_loc = face_locations(X_img)
        cv2.imshow("Images", X_img)
        if cv2.waitKey(1) != -1:
            cv2.destroyAllWindows()
            return
        if len(X_faces_loc) == 0:
            return []
        knn_clf = classifier
        faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_faces_loc)
        closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
        is_recognized = [closest_distances[0][i][0] <= DIST_THRESH for i in range(len(X_faces_loc))]
     
        # predict classes and cull classifications that are not with high confidence
        result = [(pred, loc) if rec else ("N/A", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_faces_loc, is_recognized)]
        return result
    
    def cluster_prediction(self, classifier, image_folder):
        relocate_folder = 'cluster'
        files = self.get_folder_contents(image_folder)
        for image_file in files:
            image_file_full_path = os.path.join(image_folder, image_file)
            if os.path.isdir(image_file_full_path):
                continue
            result = self.predict(classifier, image_file_full_path)
            if result:
                name = result[0][0]
                if name == self._label_name:
                    new_location = os.path.join(image_folder, relocate_folder, image_file)
                    os.rename(image_file_full_path, new_location)
            
        


# In[ ]:


print("Starting...")
ob = ImageRecog(label_name='Actor')
clf = ob.train(r"E:\Downloads\Training Stuff\Video Training\Training")
ob.cluster_prediction(clf, r"E:\Downloads\")

