# Yoga Pose Classification with MoveNet on DepthAI

This demo demonstrates the classification of poses.

## Install

Install the required packages with the following command:

```angular2html
python3 -m pip install -r requirements.txt
```

## Usage

For host mode

-> python3 demo.py

For edge mode

-> python3 demoe.py


## CSV Folder 

The ```fitness_poses_csvs_out_processed_f ``` contains csv files, one for each pose. It contains information on the sample images of each class. The first column is the image name, and next there are x and y coordinates corresponding to each joint point for that image, nose, left eye, right eye and so on. It contains x and y coordinates for 17 joint keypoints.
 To create more poses, change line 46 of data_col.py to the name of the new pose.

# MCP
