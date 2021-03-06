from math import sqrt
from sklearn import neighbors
from os import listdir, remove, path, makedirs, rename
from os.path import isdir, join, isfile, splitext
import pickle
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import face_recognition
from face_recognition import face_locations
from face_recognition.cli import image_files_in_folder
import cv2
import argparse
import time
import sys
from video_thread import VideoCaptureThread
import operator
import shutil


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


class FaceRecognizer():
    def __init__(self, training_dir = None, model_path = "", test_image_store_location = None):
        self._model_path = model_path
        self._training_dir = training_dir
        self._predictor = None
        self._temp_file = 'temp.jpg'
        self._image_data = None
        self._test_image_store_location = test_image_store_location
        self._occurance_in_video = dict()
        self.cascade_path = "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(self.cascade_path)
        
    
    def exit_program(self, message, error_code):
        print(message)
        print('Aborting Program')
        sys.exit(error_code)
    
    def save_image_store(self,X, y, traversed_images, file_name):
        print("Storing Image data...")
        images_data = (X, y, traversed_images)
        with open(file_name, 'wb') as f:
            pickle.dump(images_data, f)
    
        self._image_data = images_data
    
    def train(self, n_neighbors = None, knn_algo = 'ball_tree', 
            image_face_only = False, sample_size = 200, image_store = 'pickled_images.model', 
            load_saved_images = False, delete_missing_files = False, save_steps = 500, update_training_data = False):
        
        X = []
        y = []
        traversed_images = []
        train_dir = self._training_dir
        model_save_path = self._model_path
        image_data_save_steps = save_steps

        # Load pickled training data. Storing pickled training data save time by loading data instead of retraining classifier.
        if load_saved_images:
            # If pickled file exists then load training data
            if path.exists(image_store):
                with open(image_store, 'rb') as f:
                    X, y, traversed_images = pickle.load(f)
            else:
                self.exit_program("Image store not found...", 1)
        else:
            # If pickled data exists and load_saved_images = False then the pickled file will be overwitten. 
            if path.exists(image_store):
                self.exit_program("Please remove the existing image store and run the program again. Check in place to prevent accidental image store overwrites", 1)

        if update_training_data:
            for class_dir in listdir(train_dir):
                print("Training for %s" % class_dir)            
                if not isdir(join(train_dir, class_dir)):
                    continue

                image_list = image_files_in_folder(join(train_dir, class_dir))
                pre_loaded_image_data = [ [index,image] for index,image in enumerate(traversed_images) if y[index] == class_dir ]
                if len(pre_loaded_image_data) != len(image_list):
                    if load_saved_images and delete_missing_files:    
                        shifter = 0
                        image_name_list = [ path.split(name)[1] for name in image_list ]
                        for data in pre_loaded_image_data:
                            if data[1] not in image_name_list:
                                print("Image Not Found : %s at index %d" % (data[1],data[0]))
                                print("Deleting from image store...")
                                traversed_images.pop(data[0] - shifter)
                                X.pop(data[0] - shifter)
                                y.pop(data[0] - shifter)
                                shifter = shifter + 1               
                    
                        self.save_image_store(X, y, traversed_images, image_store)
                        
                    for img_path in image_list:
                        try:
                            image_name = path.split(img_path)[1]
                            if image_name in traversed_images:
                                continue
                            image = face_recognition.load_image_file(img_path)
                            if not image_face_only:
                                faces_bboxes = face_locations(image)
                                if len(faces_bboxes) != 1:
                                    remove(img_path)
                                    print("image {} not fit for training: {}".format(img_path, "didn't find a face" if len(faces_bboxes) < 1 else "found more than one face"))
                                    continue

                                X.append(face_recognition.face_encodings(image, known_face_locations=faces_bboxes)[0])
                            
                            else:
                                face_box = image.shape
                                box = [( 0, face_box[0], face_box[1], 0 )]
                                X.append(face_recognition.face_encodings(image, known_face_locations = box )[0])

                            y.append(class_dir)
                            traversed_images.append(image_name)
                            if image_data_save_steps == 0:
                                self.save_image_store(X, y, traversed_images, image_store)
                                image_data_save_steps = save_steps
                            else:
                                image_data_save_steps = image_data_save_steps - 1

                        except Exception:
                            continue
                
        if n_neighbors is None:
            n_neighbors = int(round(sqrt(len(X))))
            print("Chose n_neighbors automatically as:", n_neighbors)

        knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')

        self.save_image_store(X, y, traversed_images, image_store)
        
        
        knn_clf.fit(X, y)
        
        if model_save_path != "":
            with open(model_save_path, 'wb') as f:
                pickle.dump(knn_clf, f)
        
        return knn_clf

    def load_predictor(self, knn_clf = None):
        if knn_clf is None:
            model_save_path = self._model_path
        else:
            self._predictor = knn_clf
        
        if knn_clf is None and model_save_path == "":
            raise Exception("must supply knn classifier either thourgh knn_clf or model_save_path")

        if knn_clf is None:
            with open(model_save_path, 'rb') as f:
                knn_clf = pickle.load(f)
    
    def predict_from_cv_read(self, image, DIST_THRESH = .5):
        
        X_img = image
        X_faces_loc = []

        # image[ b[0]:b[2], b[3]:b[1]]) 
        # gray_image[y: y + h, x: x + w] 
        X_faces_loc = face_locations(X_img)
        '''
        try:
            faces = self.face_cascade.detectMultiScale(X_img, minNeighbors = 20, minSize = (80,80))
        except Exception:
            return []
        for (x,y,w,h) in faces:
            X_faces_loc.append([ y, x+w, y+h, x ])    
        '''
      
        if len(X_faces_loc) == 0:
            return []

        knn_clf = self._predictor
        faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_faces_loc)
        closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
        is_recognized = [closest_distances[0][i][0] <= DIST_THRESH for i in range(len(X_faces_loc))]
        # predict classes and cull classifications that are not with high confidence
        result = [(pred, loc) if rec else ("N/A", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_faces_loc, is_recognized)]
        return result
    
    def predict_from_file(self, X_img_path, DIST_THRESH = .5):
        
        if not isfile(X_img_path) or splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
            raise Exception("invalid image path: {}".format(X_img_path))

        X_img = face_recognition.load_image_file(X_img_path)
        X_faces_loc = face_locations(X_img)
        if len(X_faces_loc) == 0:
            return []

        knn_clf = self._predictor
        faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_faces_loc)
        closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
        is_recognized = [closest_distances[0][i][0] <= DIST_THRESH for i in range(len(X_faces_loc))]
        # predict classes and cull classifications that are not with high confidence
        result = [(pred, loc) if rec else ("N/A", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_faces_loc, is_recognized)]
        return result

       
    def video_occurance_count(self, name, ignore = 'ignore'):
        if name == ignore or name == 'unknown':
            return
        try:
            self._occurance_in_video[name] = self._occurance_in_video[name] + 1
        except Exception:
            self._occurance_in_video[name] = 1
    
    def get_highest_occurance_count_in_video(self):
        try:
            name = max(self._occurance_in_video, key=self._occurance_in_video.get) 
        except Exception:
            return 'None',0
        return name, self._occurance_in_video[name]

    def get_occurance_in_video(self):
        return self._occurance_in_video

    def reset_occurance(self):
        self._occurance_in_video = dict()
    
    def predict_in_video(self, video_path, show_video = True, frame_skip_number = 50.0, 
                            save_test_frame = False, highlight_face = False, scaling = 0, callback = None ):

        print("Video Path : %s" % video_path)
        if not path.exists(video_path):
            return
        temp_file = self._temp_file
        video_filename = path.split(video_path)[1]
        self.reset_occurance()
        video_player = VideoCaptureThread(video_path, frame_skip_number, size = scaling)
        video_player.start()
        success, read, frame, buffer = video_player.play_video()
        frame = 0
        time.sleep(5)
        while success:
            '''
            if not cv2.imwrite(self._temp_file, read):
                continue
            '''
            preds = self.predict_from_cv_read(read)
            
            frame = frame + frame_skip_number
            if len(preds) > 0:
                if save_test_frame:
                    for pred in preds:
                        pred_name = 'unknown' if pred[0] == 'N/A' else pred[0]
                        self.video_occurance_count(pred_name)
                        ac_name, occurance = self.get_highest_occurance_count_in_video()
                        print("Face detected\t: %20s\t\tHighest Count\t: %s (%d)" % (pred_name, ac_name, occurance ) )
                        face_loc = pred[1]
                        test_file_name = pred_name + '_' + video_filename + '_frame_' + str(int(frame)) + '.jpg'
                        directory = path.join(self._test_image_store_location, pred_name)
                        test_file_full_path = join(self._test_image_store_location, test_file_name)
                        
                        '''
                        if not path.exists(directory):
                            makedirs(directory)
                        '''
                        
                        if pred_name != 'unknown' and pred_name != 'ignore':
                            cv2.imwrite(test_file_full_path, read[ face_loc[0]:face_loc[2], face_loc[3]:face_loc[1]] )

                                          
                        if highlight_face and show_video and (pred_name != 'unknown' and pred_name != 'ignore'):
                            cv2.putText(read, pred_name, (face_loc[3], face_loc[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
                            cv2.rectangle(read, (face_loc[3],face_loc[0]), (face_loc[1],face_loc[2]),(0,255,0),2)

                        if callback(ac_name, occurance, video_path):
                            video_player.release_resource()
                            video_player.end_process(None, None)
                            video_player.join()
                            folder, filename = path.split(video_path)
                            folder_path, folder_name = path.split(folder)
                            new_folder = path.join(folder, pred_name)
                            new_file = path.join(new_folder, filename)
                            print("Identified following in the video : %s" % filename )
                            occ = self.get_occurance_in_video()
                            for key in occ:
                                print("Actor\t: %20s\t\tOccurance\t: %s" % (key, occ[key] ) )
                            if not path.exists(new_folder):
                                makedirs(new_folder)
                            try:
                                rename(video_path, new_file)
                                print("INFO : File moved")
                            except PermissionError:
                                print("ERROR : File used by another process. Cannot move")
                                
                            return
                        

            

            if show_video:
                cv2.putText(read, str(frame), (1,30), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
                cv2.putText(read, str(buffer), (1,80), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
                cv2.imshow("Video", read)
                if cv2.waitKey(5) != -1:
                    cv2.destroyAllWindows()
                    return

            success, read, frame, buffer = video_player.play_video()
            #print("Frame : %d" % frame)
            if frame > 30000:
                print("DEBUG : Frame limit exceeded...")
                break

        try:
            print("No name found...")
            dest_folder = 'Unmarked'
            video_player.release_resource()
            video_player.end_process(None, None)
            folder, filename = path.split(video_path)
            dest_full_path = path.join(folder, dest_folder, filename)
            print("INFO : Waiting for thread to end.")
            video_player.join()
            rename(video_path, dest_full_path)
            print("INFO : File moved")
        except PermissionError:
            print("ERROR : File used by another process. Cannot move")


def pred_callback(*args):
    name, occurance, video_path = args
    folder, filename = path.split(video_path)
    folder_path, folder_name = path.split(folder)
    
    new_folder = path.join(folder, name)
    new_file = path.join(new_folder, filename)
    if occurance >= 5:
        return True
    else:
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Arguments for Face Recognition')
    parser.add_argument('--train_dir', action = 'store', dest = 'train_dir', required = True)
    parser.add_argument('--video_path', action = 'store', dest = 'video_path', required = True)
    parser.add_argument('--test_dir', action = 'store', dest = 'test_dir', required = True)
    parser.add_argument('--image_store', action = 'store', dest = 'image_store', required = False)
    parser.add_argument('--model_path', action = 'store', dest = 'model_path', required = False)
    face_args = parser.parse_args()
    train_dir = face_args.train_dir
    video_path = face_args.video_path
    test_dir = face_args.test_dir
    image_store = face_args.image_store
    model_path = face_args.model_path
    update_data = False
    show_video = False
    frame_skip_number = 100.0
    resolution = 1080
    
    ob = FaceRecognizer(training_dir = train_dir, test_image_store_location = test_dir, model_path = 'new_recog.model')
    clf = ob.train(image_face_only = True, load_saved_images = True, delete_missing_files = update_data, update_training_data = update_data)
    ob.load_predictor(knn_clf = clf)
    
    
    if isdir(video_path):
        files = listdir(video_path)
        for file in files:
            video_file = path.join(video_path, file)
            if isdir(video_file):
                continue
            ob.predict_in_video(video_file, show_video = show_video, frame_skip_number = frame_skip_number, save_test_frame = True, 
                                highlight_face = True, scaling = resolution, callback = pred_callback) 
    else:
        frame_skip_number = 50
        ob.predict_in_video(video_path, show_video = show_video, frame_skip_number = frame_skip_number, save_test_frame = True, 
                            highlight_face = True, scaling = resolution, callback = pred_callback)
    