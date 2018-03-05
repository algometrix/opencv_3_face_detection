from video_thread import VideoCaptureThread
import cv2, time

video_player = VideoCaptureThread(r"F:\Holding\New Folder\Pending\naughtyamerica.17.12.25.kayla.kayden.4k.mp4", 1)
video_player.start()
success, read = video_player.play_video()
while success:
    success, read = video_player.play_video()
    cv2.imshow("Video", read)
    time.sleep(0.1)
    if cv2.waitKey(1) != -1:
        cv2.destroyAllWindows()
        