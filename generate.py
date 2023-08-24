import os
import cv2
import json
import numpy as np
from paddleocr import PaddleOCR
from uuid import uuid4


OCR_ENGINE = PaddleOCR(lang = "en", show_log = False)
HOST = "localhost"
PORT = 9000


def get_region_id():
    """
    Generates a UUID to be used as a Region ID.
    
    Returns:
    --------
        - `str`: 36 characters UUID
    """
    return str(uuid4()).replace("-", "")


def _total_files(parent_directory: str):
    """
    Get the total number of files that are to be
    scanned.
    
    Args:
    -----
        - `parent_directory`: Directory path containing all the files
    
    Returns:
    --------
        - `int`: Total files to be scanned
    """
    count = 0
    for _, _, files in os.walk(parent_directory):
        count += len(files)
    return count

    
def get_directory_tree(parent_directory: str):
    """
    Gets the Generator object that returns paths of
    all children files contained in a directory.
    
    Args:
    -----
        - `parent_directory`: Path to Images Directory
        
    Returns:
    --------
        - `str`: Generator that returns the file paths
        of the images.
    """
    return (os.path.join(root, file) for root, _, files in os.walk(parent_directory) for file in files)
           
           
def get_file_URL(filepath: str):
    """
    Converts the physical file path into a URL
    
    Args:
    -----
        - `filepath`: filepath of the image
        
    Returns:
    --------
        - `str`: URL for the file
    """
    return f"http://{HOST}:{PORT}/{filepath}".replace("\\", "/")
  
  
def read_image(filepath: str) -> np.ndarray:
    """
    Reads the image using OpenCV and returns the 
    image along with the dimensions
    
    Args:
    -----
        - `filepath`: filepath to the image.
    
    Returns:
    --------
        - `ndarray`: OpenCV image
    """         
    return cv2.imread(filepath, cv2.IMREAD_ANYCOLOR)
    
    
def OCR_Parser(image: np.ndarray):
    """
    Gets the Text and Bounding Box data for input image
    using Paddle OCR. A `Generator` object is used so conserve
    system memory instead of returnin the entire list.
    
    Args:
    -----
        - `image`: OpenCV image to be parsed by OCR Engine
        
    Returns:
    --------
        - `Generator` containing the following unpackable items:
            - `region_id`: (str) Unique ID of the bounding box
            - `bottom_left`: (list) Co-Ordinates of Bottom Left corner of Bounding Box
            - `upper_right`: (list) Co-Ordinated of Upper Right corner of Bounding Box
            - `text`: (str) Text Extracted
            
    Examples:
    ---------
    
    ```python
    >>> image = read_image(path_to_image)
    >>> for reg_id, bl, ur, text in OCR_Parser(image):
    ...     print(reg_id, bl, ur, text)
        
    ```
    """
    results = OCR_ENGINE.ocr(image, cls = False)[0]
    
    # loop over each region
    for item in results:
        if not item[1][0]:
            continue
        
        yield get_region_id(), item[0][0], item[0][2], item[1][0]


def convert_bounding_box_format(bottom_left: list, upper_right: list):
    """
    Converts the Bounding Box of `[x_min, y_min, x_max, y_max]` to `[x, y, w, h]`
    
    Args:
    -----
        - `bottom_left`: list containing `[x_min, y_min]`
        - `upper_right`: list containing `[x_max, y_max]`
        
    Returns:
    --------
        - `list`: containing `[x, y, width, height]`
    """       
    # unpack co-ordinates
    x_min, y_min = bottom_left
    x_max, y_max = upper_right

    # get the height and width
    heigth = y_max - y_min
    width  = x_max - x_min

    return [x_min, y_min, width, heigth]


def normalize_bbox(bbox: list, image_dim: tuple):
    """
    Converts bounding boxes from whole values to
    percent values
    
    Args:
    -----
        - `bbox`: List of Bounding Box Co-Ordinates
        - `image_dim`: Dimensions of the image
        
    Returns:
    --------
        - `list`: Normalized Bounding Boxes
    """
    # unpack the values
    image_width, image_height = image_dim
    x, y, w, h = bbox
    
    # conversion
    x = x / image_width * 100
    y = y / image_height * 100
    w = w / image_width * 100
    h = h / image_height * 100
    
    return [x, y, w, h]


def export_label_studio_task(image_directory: str, output_filepath: str, include_bbox = True, include_transcript = True, include_labels = True):
    
    label_studio_tasks = list()
    _file_count = _total_files(image_directory)
    
    for image_path in get_directory_tree(image_directory):
        # check if the file is an image
        if not image_path.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        
        # read the image
        image = read_image(image_path)
        
        # get image dimensions
        image_width, image_height, _ = np.shape(image)
        
        # containers to store results
        annotation_result = list()
        output_json = dict()
        
        # parse image through OCR engine
        for region_id, bottom_left, upper_right, text in OCR_Parser(image):
            
            # get bounding box co-ordinates
            xywh_bbox = convert_bounding_box_format(bottom_left, upper_right)
            x, y, w, h = normalize_bbox(xywh_bbox, (image_width, image_height))
            
            # different results
            if include_bbox:
                bbox = {
                    "original_width": image_width,
                    "original_height": image_height,
                    "image_rotation": 0,
                    "value": {"x": x, "y": y, "width": w, "height": h, "rotation": 0},
                    "id": region_id,
                    "from_name": "bbox",
                    "to_name": "image",
                    "type": "rectangle"
                }
                annotation_result.extend([bbox])
                
            if include_transcript:
                transcript = {
                    "original_width": image_width,
                    "original_height": image_height,
                    "image_rotation": 0,
                    "value": {"x": x, "y": y, "width": w, "height": h, "rotation": 0, "text": [text]},
                    "id": region_id,
                    "from_name": "transcription",
                    "to_name": "image",
                    "type": "textarea"
                }
                annotation_result.extend([transcript])
                
            if include_labels:
                label = {
                    "original_width": image_width,
                    "original_height": image_height,
                    "image_rotation": 0,
                    "value": {"x": x, "y": y, "width": w, "height": h, "rotation": 0, "labels": ['TEXT']},
                    "id": region_id,
                    "from_name": "label",
                    "to_name": "image",
                    "type": "labels"
                }
                annotation_result.extend([label])
        
        output_json['data'] = {"ocr": get_file_URL(image_path)}
        output_json['predictions'] = [{"result": annotation_result, "score": 0.9}]
        
        label_studio_tasks.append(output_json)
        
    with open(output_filepath, 'w') as f:
        json.dump(label_studio_tasks, f, indent=4)
            
               
           
           
if __name__ == "__main__":
    export_label_studio_task("images", "label-studio-task.json", include_labels = False)