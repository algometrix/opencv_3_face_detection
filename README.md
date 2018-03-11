# opencv_3_face_detection
Face recognition in videos using face_recognition (dlib) and OpenCV 3.


# Installing Requirements and setting up the enivornment

1. Download Anaconda Python 3.6 version from https://www.anaconda.com/download/
2. Clone this repository.
3. Goto repository folder and run `conda env create -f environment.yml`. This will take some time, so please be patient.
4. After the environment is ready, run `activate face_detect` for Windows or `source activate face_detect` for MacOS or Linux.

# Running the demo

1. Create two directories for training data and output data.
2. Create another directoy **inside** the new training directory and name it with the desired lable. e.g Actor name if it contains actor's training images.
3. Place the training images inside this folder.
4. Run the face detection and recognition script using `python face_detect.py --train_dir "E:\KNN Train" --test_dir "E:\Test Data" --video_path "S:\Video\file.mp4"` where `E:\KNN Train` contains training data, `E:\Test Data` is a blank directory to store face detected in the video file and `S:\Video\file.mp4` is the path of the video file.


# Tools Used
1. Visual Studio Code
2. Anaconda Python 3.6 version
3. Github Desktop for Windows 