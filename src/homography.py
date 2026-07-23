import cv2 
from .court_tempelete import build_court_template 
import numpy as np 

def compute_homography(
    src_pts,
    dst_pts ,
    min_points = 4,
    ransac_thresh = 5.0
) :

    if src_pts is None or len(src_pts) < min_points:
        return None, None

    H, mask = cv2.findHomography(
        src_pts, dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh,
    )
    return H, mask


def transform_ball_position(keypoints , ball_positon) :
    #             kp_world = np.array([
    #         dtl, dtr, dbr, dbl,stl, str_,sbr,sbl,svtl ,svtr , svbr , svbl  , svc_mid_t , svc_mid_b 
    # ], dtype=np.float32)
    court_template, kp_world = build_court_template()
    dtl , dtr , dbl , dbr , stl , sbl , str_ , sbr , svtl , svtr , svbl , svbr , svc_mid_t ,svc_mid_b = keypoints
    kp_frame = np.array([dtl , dtr ,dbr ,  dbl  , stl , str_ ,sbr,  sbl   , svtl ,
                          svtr , svbr, svbl , svc_mid_t ,svc_mid_b] , dtype=np.float32) 
    H, mask = compute_homography(kp_frame, kp_world)
    ball = np.array([ball_positon] ,dtype=np.float32 ) 
    if H is not None:
        proj = cv2.perspectiveTransform(
            ball.reshape(-1, 1, 2), H).reshape(-1, 2)
            
    return proj[0]  