from video_thread import VideoCaptureThread
import cv2, time
import sys



if __name__ == '__main__':
    video_path = sys.argv[1]
    video_player = VideoCaptureThread(video_path, 1)
    video_player.start()
    success, read = video_player.play_video()
    while success:
        success, read = video_player.play_video()
        cv2.imshow("Video", read)
        time.sleep(0.02)
        if cv2.waitKey(1) != -1:
            cv2.destroyAllWindows()

        