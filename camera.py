# Let us import the Libraries required.
import cv2
import numpy as np

from Pipeline import *
from stress_detection_stages import *

class VideoCamera(object):

    """ Takes the Real time Video, Predicts the Emotion using pre-trained model. """

    def __init__(self):
        self.video = cv2.VideoCapture(0)

        self.stress_detection_pipeline = SequentialPipeline()
        self.setup_pipeline()
        self.stress_detection_pipeline.execute_setup()

    def __del__(self):
        self.video.release()

    def setup_pipeline(self):
        self.stress_detection_pipeline.add_setup_stage('LOAD_MODELS', LoadModels())
        self.stress_detection_pipeline.add_stage('GRAB_FRAME', GrabFrame())
        self.stress_detection_pipeline.add_stage('FACE_DETECTION', FaceDetection())
        self.stress_detection_pipeline.add_stage('EMOTION_DETECTION', EmotionDetection())
        self.stress_detection_pipeline.add_stage('STRESS_CLASSIFIER', StressClassifier())
        self.stress_detection_pipeline.add_stage('GENERATE_DISPLAY_FRAME', GenerateDisplayFrame())

        self.stress_detection_pipeline.static_io.model_files = {
          'face_detection_model': 'haarcascade_frontalface_default.xml',
          'emotion_detection_model': ('model.json', 'model_weights.h5'),
          'stress_classifier_model': 'random_forest_model.sav'
        }

    def get_frame(self):
        """It returns camera frames along with bounding boxes and predictions"""

        # Reading the Video and grasping the Frames
        _, frame = self.video.read()

        self.stress_detection_pipeline.io.frame = frame
        outcome = self.stress_detection_pipeline.execute()

        if outcome == status.SUCCESS:
            return self.stress_detection_pipeline.io.display_frame_bytes
        elif outcome == status.FAILURE:
            _, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()