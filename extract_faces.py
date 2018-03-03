"""
This is an example of using the k-nearest-neighbors(knn) algorithm for face recognition.

When should I use this example?
This example is useful when you whish to recognize a large set of known people,
and make a prediction for an unkown person in a feasible computation time.

Algorithm Description:
The knn classifier is first trained on a set of labeled(known) faces, and can then predict the person
in an unkown image by finding the k most similar faces(images with closet face-features under eucledian distance) in its training set,
and performing a majority vote(possibly weighted) on their label.
For example, if k=3, and the three closest face images to the given image in the training set are one image of Biden and two images of Obama, 
The result would be 'Obama'.
*This implemententation uses a weighted vote, such that the votes of closer-neighbors are weighted more heavily.

Usage:
-First, prepare a set of images of the known people you want to recognize.
 Organize the images in a single directory with a sub-directory for each known person.
-Then, call the 'train' function with the appropriate parameters.
 make sure to pass in the 'model_save_path' if you want to re-use the model without having to re-train it. 
-After training the model, you can call 'predict' to recognize the person in an unknown image.

NOTE: This example requires scikit-learn to be installed! You can install it with pip:
$ pip3 install scikit-learn
"""

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
import time
import shutil


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
FACE_LOC = r"E:\Downloads\NewFolder\Facial Features"

def train(train_dir, model_save_path = "", n_neighbors = None, knn_algo = 'ball_tree', verbose=False, load_model = False):
    """
    Trains a k-nearest neighbors classifier for face recognition.

    :param train_dir: directory that contains a sub-directory for each known person, with its name.

     (View in source code to see train_dir example tree structure)

     Structure:
        <train_dir>/
        ├── <person1>/
        │   ├── <somename1>.jpeg
        │   ├── <somename2>.jpeg
        │   ├── ...
        ├── <person2>/
        │   ├── <somename1>.jpeg
        │   └── <somename2>.jpeg
        └── ...
    :param model_save_path: (optional) path to save model of disk
    :param n_neighbors: (optional) number of neighbors to weigh in classification. Chosen automatically if not specified.
    :param knn_algo: (optional) underlying data structure to support knn.default is ball_tree
    :param verbose: verbosity of training
    :return: returns knn classifier that was trained on the given data.
    """
    X = []
    y = []
    if verbose:
        print("Training")
    for class_dir in listdir(train_dir):
        if verbose:
            print("Finding Faces in %s" % class_dir)
        if not isdir(join(train_dir, class_dir)):
            continue
        for img_path in image_files_in_folder(join(train_dir, class_dir)):
            new_img = img_path.replace(train_dir, FACE_LOC)
            print("Face Extracted")
            folder_path, renamed_file = path.split(new_img)
            full_path, folder_name = path.split(folder_path)
            renamed_file = folder_name.lower().replace(' ','_') + '_'+ renamed_file
            new_img = join(folder_path, renamed_file)
            print(new_img)
            if path.exists(new_img):
                continue
            
            try:
                image = face_recognition.load_image_file(img_path)
                faces_bboxes = face_locations(image)
            except Exception:
                continue
            
            if len(faces_bboxes) != 1:
                remove(img_path)
                if verbose:
                    print("image {} not fit for training: {}".format(img_path, "didn't find a face" if len(faces_bboxes) < 1 else "found more than one face"))
                continue
            
            
            folder, file_name = path.split(new_img)
            if not path.exists(folder):
                makedirs(folder)
            
            b = faces_bboxes[0]
            cv2.imwrite(new_img, (image[ b[0]:b[2], b[3]:b[1]]) )
            X.append(face_recognition.face_encodings(image, known_face_locations=faces_bboxes)[0])
            #X.append(face_recognition.face_encodings(image, known_face_locations=[ (0, 0, image.shape[0], image.shape[1] )] )[0] )
            y.append(class_dir)


    if n_neighbors is None:
        n_neighbors = int(round(sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically as:", n_neighbors)

    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')

    if verbose:
        print("Fitting data...")
    knn_clf.fit(X, y)
    if verbose:
        print("Fitting completed...")

    if model_save_path != "":
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)
    return knn_clf

def predict(X_img_path, knn_clf = None, model_save_path ="", DIST_THRESH = .5):
    """
    recognizes faces in given image, based on a trained knn classifier

    :param X_img_path: path to image to be recognized
    :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
    :param model_save_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
    :param DIST_THRESH: (optional) distance threshold in knn classification. the larger it is, the more chance of misclassifying an unknown person to a known one.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'N/A' will be passed.
    """

    if not isfile(X_img_path) or splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("invalid image path: {}".format(X_img_path))

    if knn_clf is None and model_save_path == "":
        raise Exception("must supply knn classifier either thourgh knn_clf or model_save_path")

    if knn_clf is None:
        with open(model_save_path, 'rb') as f:
            knn_clf = pickle.load(f)

    X_img = face_recognition.load_image_file(X_img_path)
    X_faces_loc = face_locations(X_img)
    if len(X_faces_loc) == 0:
        return []

    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_faces_loc)


    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)

    is_recognized = [closest_distances[0][i][0] <= DIST_THRESH for i in range(len(X_faces_loc))]

    # predict classes and cull classifications that are not with high confidence
    return [(pred, loc) if rec else ("N/A", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_faces_loc, is_recognized)]

def draw_preds(img_path, preds):
    """
    shows the face recognition results visually.

    :param img_path: path to image to be recognized
    :param preds: results of the predict function
    :return:
    """
    source_img = Image.open(img_path).convert("RGBA")
    draw = ImageDraw.Draw(source_img)
    for pred in preds:
        loc = pred[1]
        name = pred[0]
        # (top, right, bottom, left) => (left,top,right,bottom)
        draw.rectangle(((loc[3], loc[0]), (loc[1],loc[2])), outline="red")
        #draw.text((loc[3], loc[0] - 30), name, font=ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 30))
    source_img.show()

def play_video_in_cv2(video_path, highlight_faces = True):
    print("Video Path : %s" % video_path)
    temp_file = 'temp.jpg'
    vc = cv2.VideoCapture(video_path)
    fps = vc.get(cv2.CAP_PROP_FPS)
    size = (int(vc.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    success, read = vc.read()
    i = 0
    count = 0
    while success:
        success, read = vc.read()
        i = i + 200.0
        vc.set(cv2.CAP_PROP_POS_FRAMES, i )
        cv2.imwrite(temp_file, read)
        preds = predict(temp_file , model_save_path = 'faces_folder_names.model')
        print(preds)
        #cv2.imshow("Video", read)
        if cv2.waitKey(1) != -1:
            vc.release()
            cv2.destroyAllWindows()
            return

if __name__ == "__main__":
    knn_clf = train(r"E:\Downloads\NewFolder\New Files", model_save_path = 'larger_set.model', verbose=True)
    #test_path = r""
    #play_video_in_cv2(test_path)
    
    '''image = cv2.imread(join(test_path, img_path))
    full_path = join(test_path, img_path)
    preds = predict(full_path , model_save_path = 'faces_folder_names.model')
    cv2.imshow("Video", image)
    cv2.waitKey(1)
    draw_preds(join(test_path, img_path), preds)'''
    
    


    

    
    
    

