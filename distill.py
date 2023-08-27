"""
Module: distill.py

Description
-----------
This module is to be used after the user has completed their annotation job
on Label Studio. The user is expected to 
"""

import json
import yaml
import os
from urllib.parse import unquote


LABEL_STUDIO_JSON_PATH = "project-15-at-2023-08-26-15-57-f5ec33aa.json"
DISTILLED_ANNOTATIONS_PATH = "layoutlm-annotations.json"


def import_annotations(filepath: str):
    with open(filepath, "r") as fp:
        annotations = json.load(fp)
    return (annotation for annotation in annotations)


def get_label_id_map() -> dict:
    with open("labels.yml", "r") as fp:
        labels = yaml.load(fp, yaml.SafeLoader)
        labels = labels['labels']
        print(labels)
        
    return {label: idx for idx, label in enumerate(labels)}
    
    
def parse_annotation(annotation_dict: dict):
    # get the image url
    image_url = annotation_dict['ocr']
    
    # get bounding boxes
    bbox = [[box['x'], box['y'], box['width'], box['height']] for box in annotation_dict['bbox']]
    
    # get the transcriptions
    text = annotation_dict['transcription']
    
    # get the labels
    labels = [box['labels'][0] for box in annotation_dict['label']]
    
    # count the anntations
    count = len(annotation_dict['bbox'])
    
    return image_url, bbox, text, labels, count


def denormalize_bbox(bbox: list):
    return [int(10 * point) for point in bbox]


def get_image_filepath(image_url: str):
    # clean the 
    decoded_url = unquote(image_url)
    cleaned_url = decoded_url.replace('\\', '/')
    
    # get the filename
    filename = os.path.basename(cleaned_url)
    system_filepath = f"/images/{filename}"
    
    return system_filepath


def distill_annotation(image_filepath:str, bbox:list, text:list, labels:list):
    return dict(image = image_filepath, bbox = bbox, words = text, word_labels = labels)


def export_annotations(annotations: list, output_filepath: str):
    with open(output_filepath, "w") as fp:
        json.dump(annotations, fp, indent = 5)

    
def main(annotations_filepath, output_filepath):
    annotations = import_annotations(annotations_filepath)
    label2id = get_label_id_map()
    exportable_annotations = list()
        
    # loop over the annotations
    for idx, annotation in enumerate(annotations, 1):
        image_url, bbox_norm, text, labels, count = parse_annotation(annotation)
        
        # denormalize the bounding boxes
        bbox_denorm = [[denormalize_bbox(box)] for box in bbox_norm]
        
        # convert labels to label ids
        label_ids = [label2id[label] for label in labels]
        
        # convert the url of image to image filepath
        image_filepath = get_image_filepath(image_url)
        
        # prepare the distilled annotation dict
        distilled_annotation = distill_annotation(image_filepath, bbox_denorm, text, label_ids)
        
        # append to main list
        exportable_annotations.append(distilled_annotation)
        
        print(f"[{idx}] Processed: {image_filepath}, containing {count} annotations")
        
    # export to disk
    export_annotations(exportable_annotations, output_filepath)
    

if __name__ == "__main__":
    main(LABEL_STUDIO_JSON_PATH, DISTILLED_ANNOTATIONS_PATH)