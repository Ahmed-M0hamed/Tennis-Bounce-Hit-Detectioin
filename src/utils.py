import cv2
from typing import List  , Tuple
import json 
import numpy as np 

def get_bottom_center_of_player(points : List ) -> Tuple : 
    x1 , y1 , x2, y2 = points 
    x_center = int(x1 + (x2-x1  )/2 ) 
    y_center = int(y1-(y1-y2)) 
    return (x_center, y_center) 
 
def read_frames(path) : 
    frames = [] 
    cap = cv2.VideoCapture(path) 
    fps = cap.get(cv2.CAP_PROP_FPS)

    while True : 
        ret , frame = cap.read()  
        if not ret : 
            break 
        frames.append(frame) 
    cap.release() 
    return frames  ,fps 

def write_video(  output_path , frames , fps  ) : 
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc,fps , (frames[0].shape[1], frames[0].shape[0]))
        for frame in frames :
            out.write(frame)
        out.release()

def read_annoations(path : str ) -> List : 
    with open(path , 'rb') as f :
        annotations = json.load(f) 
    return annotations


def write_labels(results:List , output_path:str)->None : 
    with open(output_path , 'w',encoding="utf-8") as f : 
        json.dump(results , f , indent=4)  

def find_gaps(annotations : List ) -> List  :
    gap_start = 0 
    index = 0 
    gaps = []
    while index < len(annotations): 
        if 'ball_position' in annotations[index] : 
            gap_end = annotations[index]['frame_id'] 
            if gap_end - gap_start > 1 : 
                gaps.append((gap_start , gap_end)) 
            gap_start = gap_end 
        index += 1 
    return gaps 

import pandas as pd 
def convert_annotations_into_df(annotations) : 
    data = [] 
    for annotation in annotations : 
        if 'ball_position' in annotation : 
            x , y = annotation['ball_position'] 
            data.append([annotation['frame_id'] , x , y]) 
        else : 
            data.append([annotation['frame_id'] , None , None]) 

    df = pd.DataFrame(data , columns=['frame' , 'x' , 'y'])



