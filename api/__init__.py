import cv2
from api.faces_api import FaceLocalization
__version__ = '1.0.0'


def get_face_from_file(input_filename, align_size, output_filename):
    faces = FaceLocalization.get_face_and_align(input_filename, align_size)
    for i, face in enumerate(faces):
        cv2.imwrite("{}_{}.jpg".format(output_filename, i), face)


def get_face(image, align_size, output_filename):
    faces = FaceLocalization.get_face_and_align_from_matrix(image, align_size)
    for i, face in enumerate(faces):
        cv2.imwrite("{}_{}.jpg".format(output_filename, i), face)
