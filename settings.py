import os


class PathSettings:
    PROJECT_FOLDER = os.path.join(os.path.dirname(__file__))
    DATA_FOLDER = os.path.join(PROJECT_FOLDER, "data")
    RESULT_FOLDER = os.path.join(PROJECT_FOLDER, "results")
    MODEL_FOLDER = os.path.join(PROJECT_FOLDER, "models")