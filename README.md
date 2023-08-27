# Annotation Tool for LayoutLM

## Overview

Annotating images for fine tuning the LayoutLM model requires a lot of manual work which includes the three major steps:

1. Drawing bounding boxes around the texts
2. Labelling the bounding boxes
3. Manually type in the text in those bounding boxes

There are a lot of ML assisted annotation tools in the market that can make the job much easier but they can cost the user money. Thus I present the community with an annotation tool that automates steps 1 and 3. This can be acheived using __Label Studio__ and __PaddleOCR__, both of them are Open Source but the advantage of my tool is that you can use your own Models for Text Detection and/or Text Recognition Model (you have to overwrite a method in the code)

### How it works?

1. Copy-Paste all your images into the `images` folder.
2. Run the `generate.py` script to create a __Label Studio Annotation Job__ (A JSON file). (In this step the __PaddleOCR__ model will detect the Text Bounding boxes and the Recognisied text. Thus completing steps 1 and 3, mentioned above).
3. Upload this JSON file to Label Studio to start the Annotation Job.
4. All you have to do is to label each bounding box and fix the bounding boxes and the OCR text.
5. After the Annotation Job is done, export the annotation using `JSON-MIN` export option.
6. Use `distill.py` to postprocess the annotations so that thet can be ingested by the LayoutLM model without any preporcessing. 

## Installation

```bash
$ git clone 'https://github.com/arnabd64/LayoutLM-Annotator.git'
$ cd LayoutLM-Annotator

# create a virtual environment
# using conda
$ conda create -n label python=3.10
$ conda activate label

# using venv
$ python3 -m venv label
$ source label/bin/activate

# install packages
$ pip3 install -r requirements.txt
```

Once the process is done then the annotation tool has been installed.

## Getting started with the Annotation

### STEP 1:

Put all your image data inside the `images` folder.

### STEP 2: (Optional) 

The `server.py` script will run a HTTP file server on `http://localhost:9000/` if you want to change the port from 9000 to a port of your choice replace the `PORT` variable on line 11.

### STEP 3: (Important)

Open `generate.py` and change the `ANNOTATION_NAME` variable to what to want. Run the script

### STEP 4: (Important)

Run `server.py`

### STEP 5:

Run the Label Studio server byb executing: `label-studio --data-dir ./data`

### STEP 6:

- Do the usual Sign Up and Login In stuff.
- Once Inside the application, create a project
- Give it in an appropiate name
- In `Data Import` Tab, Upload the `.json` file that was created by `generate.py`.
- In the `Annotation` Tab, choose __Optical Character Recognition__.
- Start the Project

### STEP 7:

Once the Annotation Job is done then hit the `Export` button and export it under the `JSON-MIN` option.

### STEP 8:

Open `distill.py` and the variables `LABEL_STUDIO_JSON_PATH` to the file name of the JSON export from label studio and optionally you can change the variable `DISTILLED_ANNOIATIONS` variable to change the name of the postprocessed JSON file.

Run `distill.py`

### All is Done

# Issues and Forks

If you are facing any issues with the code and is unable to solve then do raise an issue on the github repo issues section. You have a faced an issue and have modified the code to solve then you can share it with other community members and me by raising an issue and writing down the solution in the comments section.

Other members odf the community feel free to Fork this repo and do leave a star on this repo.

# Acknowledgements

1. [@AIOdysseyhub](https://www.youtube.com/@AIOdysseyhub/videos) YouTube channel for the inspiration as well as the first draft of the code.
2. [Label Studio Documentation](https://labelstud.io/templates/optical_character_recognition.html)
3. [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)