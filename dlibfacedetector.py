import cv2 as cv
import time
import dlib
import numpy as np


class FaceDetector:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()    # This is Hog face detector
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.gray = cv.COLOR_BGR2GRAY
        self.font = cv.FONT_HERSHEY_SIMPLEX

    def im_face(self, image):   # This is for static Image, your image must in the same folder of code file
        image = cv.imread(image)
        gray = cv.cvtColor(image, self.gray)
        faces = self.detector(gray)
        for face in faces:
            cv.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
            count = str(len(faces))
            cv.putText(image, "No of Faces Detected: "+count, (165, 30), self.font, 0.7, (0, 255, 0), 2)
            lm = self.predictor(gray, face)
            for i in range(68):     # or use for i in range(lm.num_parts):
                x = lm.part(i).x
                y = lm.part(i).y
                cv.circle(image, (x, y), 1, (0, 255, 255), 1)
        cv.imshow("MyImage", image)
        cv.waitKey(0)    

    def face_lm(self, frame):    # Face landmarks detection
        gray = cv.cvtColor(frame, self.gray)
        faces = self.detector(gray)
        for face in faces:
            lm = self.predictor(gray, face)
            for i in range(lm.num_parts):
                x = lm.part(i).x
                y = lm.part(i).y
                cv.circle(frame, (x, y), 1, (0, 255, 255), 2)
                cv.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
                count = str(len(faces))
                cv.putText(frame, "No of Faces Detected: "+count, (120, 30), self.font, 0.7, (0, 255, 0), 2)


def main():
    sr0 = cv.VideoCapture(0)    # This is for primary camera input
    sr1 = cv.VideoCapture("test.mp4")   # This is for video input
    video_source = sr1
    imagesource = "people.jpg"  # This is for static image input
    if video_source is None:
        stat_image = FaceDetector()
        stat_image.im_face(imagesource)
    else:
        ptime = 0
        pts = np.array([[[10, 10], [790, 10], [790, 40], [10, 40]]])
        obj = FaceDetector()    # object creation
        while video_source.isOpened():
            _, flip = video_source.read()
            frame = cv.resize(cv.flip(flip, 1), (800, 600))
            cv.fillPoly(frame, pts=pts, color=(255, 255, 255))
            current_time = time.time()
            fps = str(int(1/(current_time-ptime)))
            ptime = current_time
            obj.face_lm(frame)   # method calling
            cv.putText(frame, "FPS: "+fps, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            cv.imshow("Webcam", frame)
            if cv.waitKey(1) & 0xFF == ord("q"):
                break
        video_source.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
