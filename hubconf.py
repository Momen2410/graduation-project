from ultralytics import YOLO

def _create(path='best.pt', autoshape=True):
    """Creates a YOLOv8 model from a specified path.

    Args:
        path (str): The path to the model weights.
        autoshape (bool): Whether to automatically infer the input shape. (Unused)

    Returns:
        YOLO: The loaded YOLOv8 model.
    """
    model = YOLO(path)
    return model