import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


class FaceDetector:
    def __init__(self, type='git', device='cpu'):
        self.type = type
        self.detector = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

    def dense(self, image):
        if self.type != 'google':
            print('FaceDetector => Wrong type for dense detection!')
            exit(-1)

        results = self.detector.process(image)

        if results.multi_face_landmarks is None:
            return None

        lmks = results.multi_face_landmarks[0].landmark
        lmks = np.array(list(map(lambda l: np.array([l.x, l.y]), lmks)))
        lmks[:, 0] = lmks[:, 0] * image.shape[1]
        lmks[:, 1] = lmks[:, 1] * image.shape[0]

        return lmks
