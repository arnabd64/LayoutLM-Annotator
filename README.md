# Automated OCR Annotation Generator Application

## Overview


## Installation

```bash
$ git clone repo-link
$ cd repo

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

### STEP 2: 

Run the script: `generate.py` and wait until the job is done.

### STEP 3:

Run the script: `server.py`. This is create a local HTTP server at port 9000 that will feed the images to Label Studio

### STEP 4: 

Run Label Studio by running the command: `label-studio --data-dir ./label-studio-data`

### STEP 5:

- Do the usual Sign Up and Login In stuff.
- Once Inside the application, create a project
- Give it in an appropiate name
- In `Data Import` Tab, Upload the `.json` file that was created by `generate.py`.
- In the `Annotation` Tab, choose __Optical Character Recognition__.
- Start the Project