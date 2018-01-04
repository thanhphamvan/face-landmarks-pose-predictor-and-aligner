import dlib
import cv2
import openface

# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
predictor_model = '/home/thanhpv/workspace/pre-processing-face/models/face_landmarks.dat'

face_detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor(predictor_model)
face_aligner = openface.AlignDlib(predictor_model)


class FaceLocalization:
    @staticmethod
    def get_face_and_align_from_matrix(image_frame, aligne_size):
        detected_faces = face_detector(image_frame, 1)
        print("Found {} faces in the image frame".format(len(detected_faces)))
        faces = []
        for i, face_rect in enumerate(detected_faces):
            # Detected faces are returned as an object with the coordinates of the top, left, right and bottom edges
            print(
                "- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(),
                                                                                   face_rect.right(),
                                                                                   face_rect.bottom()))
            # Get the the face's pose pose_landmarks = face_pose_predictor(image_frame, face_rect) Use openface to
            # calculate and perform the face alignment
            aligned_face = face_aligner.align(aligne_size, image_frame, face_rect,
                                              landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
            faces.append(aligned_face)
        return faces

    @staticmethod
    def get_face_and_align(file_name, aligne_size):
        image = cv2.imread(file_name)
        return FaceLocalization.get_face_and_align_from_matrix(image, aligne_size)

    @staticmethod
    def get_face_and_align_from_video(video_file_name, aligne_size):
        cap = cv2.VideoCapture(video_file_name)
        all_faces = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                faces = FaceLocalization.get_face_and_align_from_matrix(frame, aligne_size)
                all_faces.append(faces)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        return all_faces


print("Face APIs Ready to use.")
