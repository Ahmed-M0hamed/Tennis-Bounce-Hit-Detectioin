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



def distance_between_two_points(point_1 : Tuple , point_2 : Tuple) -> float  : 
    x1 , y1 = point_1 
    x2, y2 = point_2 
    distance = np.sqrt((x2-x1) **2 + (y2-y1)**2)
    return distance 


def detect_player_hitting(near_player : List  , far_player : List , ball_pos:Tuple) -> Tuple : 
    near_player_bottom_center = get_bottom_center_of_player(near_player) 
    far_player_bottom_center = get_bottom_center_of_player(far_player) 
    players = [near_player_bottom_center , far_player_bottom_center]
    classes = ['near' , 'far']
    distances = [distance_between_two_points(ball_pos , player) for player in players ] 
    min_index = np.argmin(distances) 
    return (players[min_index] , classes[min_index] ) 