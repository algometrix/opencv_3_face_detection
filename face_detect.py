from math import sqrt
from sklearn import neighbors
from os import listdir, remove, path, makedirs
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

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


class FaceRecognizer():
    def __init__(self, training_dir = None, model_path = "", test_image_store_location = None):
        self._model_path = model_path
        self._training_dir = training_dir
        self._predictor = None
        self._temp_file = 'temp.jpg'
        self._image_data = None
        self._test_image_store_location = test_image_store_location

    
    def exit_program(self, message, error_code):
        print(message)
        print('Aborting Program')
        sys.exit(error_code)
    
    def train(self, n_neighbors = None, knn_algo = 'ball_tree', 
            image_face_only = False, sample_size = 200, image_store = 'pickled_images.model', load_saved_images = False, delete_missing_files = True):
        
        X = []
        y = []
        traversed_images = []
        train_dir = self._training_dir
        model_save_path = self._model_path
        if load_saved_images:
            if path.exists(image_store):
                with open(image_store, 'rb') as f:
                    X, y, traversed_images = pickle.load(f)
            else:
                self.exit_program("Image store not found...", 1)
        else:
            if path.exists(image_store):
                self.exit_program("Please remove the existing image store and run the program again. Check in place to prevent accidental image store overwrites", 1)


        for class_dir in listdir(train_dir):
            print("Training for %s" % class_dir)
            if not isdir(join(train_dir, class_dir)):
                continue
            for img_path in image_files_in_folder(join(train_dir, class_dir)):
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
                except Exception:
                    continue
                


        if n_neighbors is None:
            n_neighbors = int(round(sqrt(len(X))))
            print("Chose n_neighbors automatically as:", n_neighbors)

        knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')

        print("Storing Image data...")
        images_data = (X, y, traversed_images)
        with open(image_store, 'wb') as f:
            pickle.dump(images_data, f)
                
        self._image_data = images_data
        print("Completed...")
        
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
    
    def predict(self, X_img_path, DIST_THRESH = .5, save_test_image = False, file_suffix = ''):
        
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
        if False and save_test_image:
            b = X_faces_loc[0]
            pred_name = 'unknown' if result[0][0] == 'N/A' else result[0][0]
            test_file_name = pred_name + '_' + file_suffix + '.jpg'
            directory = path.join(self._test_image_store_location, pred_name)
            test_file_full_path = join(self._test_image_store_location, test_file_name)
            if not path.exists(directory):
                makedirs(directory)
            cv2.imwrite(test_file_full_path, read[ b[0]:b[2], b[3]:b[1]] )

        return result

    def predict_in_video(self, video_path, show_video = True, frame_skip_number = 50.0, save_test_frame = False, highlight_face = False ):
        print("Video Path : %s" % video_path)
        temp_file = self._temp_file
        video_filename = path.split(video_path)[1]
        vc = cv2.VideoCapture(video_path)
        fps = vc.get(cv2.CAP_PROP_FPS)
        size = (int(vc.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        success, read = vc.read()
        frame = 0
        count = 0
        while success:
            success, read = vc.read()
            frame = frame + frame_skip_number
            vc.set(cv2.CAP_PROP_POS_FRAMES, frame )
            if not cv2.imwrite(temp_file, read):
                continue
            preds = self.predict(temp_file, save_test_image = save_test_frame, 
                                    file_suffix = video_filename + '_frame_' + str(int(frame)) )
           
            if len(preds) > 0:
                #print(preds)
                if save_test_frame:
                    for pred in preds:
                        pred_name = 'unknown' if pred[0] == 'N/A' else pred[0]
                        print("Face detected : %s" % pred_name)
                        face_loc = pred[1]
                        test_file_name = pred_name + '_' + video_filename + '_frame_' + str(int(frame)) + '.jpg'
                        directory = path.join(self._test_image_store_location, pred_name)
                        test_file_full_path = join(self._test_image_store_location,pred_name, test_file_name)
                        if not path.exists(directory):
                            makedirs(directory)
                        #print("Directory : %s" % directory)
                        #print("Test File : %s" %  test_file_full_path)
                        #print('')
                        cv2.imwrite(test_file_full_path, read[ face_loc[0]:face_loc[2], face_loc[3]:face_loc[1]] )

                        if highlight_face:
                            cv2.putText(read, pred_name, (face_loc[3], face_loc[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
                            cv2.rectangle(read, (face_loc[3],face_loc[0]), (face_loc[1],face_loc[2]),(0,255,0),2)

                    
            if show_video:
                res = cv2.resize(read,None,fx=1.0, fy=1.0, interpolation = cv2.INTER_CUBIC)
                cv2.imshow("Video", res)
            if cv2.waitKey(1) != -1:
                vc.release()
                cv2.destroyAllWindows()
                return


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

    ob = FaceRecognizer(training_dir = train_dir, test_image_store_location = test_dir, model_path = 'new_recog.model')
    clf = ob.train(image_face_only = True, load_saved_images = True)
    ob.load_predictor(knn_clf = clf)
    ob.predict_in_video(video_path, show_video = True, frame_skip_number = 10.0, save_test_frame = True, highlight_face = True)
    #preds = ob.predict(r"")
    #print(preds)
