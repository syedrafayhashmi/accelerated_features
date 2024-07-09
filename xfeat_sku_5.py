import numpy as np
import os
import torch
import tqdm
import cv2
import matplotlib.pyplot as plt
from modules.xfeat import XFeat
import shutil
import time

def warp_corners_and_draw_matches(ref_points, dst_points, img1, img2):
    H, mask = cv2.findHomography(ref_points, dst_points, cv2.USAC_MAGSAC, 3.5, maxIters=1_000, confidence=0.999)
    mask = mask.flatten()

    h, w = img1.shape[:2]
    corners_img1 = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners_img1, H)

    img2_with_corners = img2.copy()
    for i in range(len(warped_corners)):
        start_point = tuple(warped_corners[i-1][0].astype(int))
        end_point = tuple(warped_corners[i][0].astype(int))
        cv2.line(img2_with_corners, start_point, end_point, (0, 255, 0), 4)

    keypoints1 = [cv2.KeyPoint(p[0], p[1], 5) for p in ref_points]
    keypoints2 = [cv2.KeyPoint(p[0], p[1], 5) for p in dst_points]
    matches = [cv2.DMatch(i, i, 0) for i in range(len(mask)) if mask[i]]

    img_matches = cv2.drawMatches(img1, keypoints1, img2_with_corners, keypoints2, matches, None,
                                  matchColor=(0, 255, 0), flags=2)

    return img_matches

xfeat = XFeat()
dataset_name = "sku100k_v2"
query_imgs_dir = f"./datasets/{dataset_name}/query_images"
class_imgs_dir = f"./datasets/{dataset_name}/reference_images"
threshold = 5
output_dir_path = f"./datasets/output/{dataset_name}/top_{threshold}"

query_img_path = os.listdir(query_imgs_dir)
class_img_path = os.listdir(class_imgs_dir)

a = time.time()
n_query_imgs = 0

for query_index, query_img in enumerate(query_img_path):
    print("Query index: ", query_index)
    image0 = cv2.imread(os.path.join(query_imgs_dir, query_img))
    
    n_class_imgs = 0
    max_matching = -1
    filenames = {}
    y = time.time()
    
    for class_img in class_img_path:
        try:
            image1 = cv2.imread(os.path.join(class_imgs_dir, class_img))
        except:
            continue
        
        mkpts_0, mkpts_1 = xfeat.match_xfeat(image0, image1, top_k=1024)
        
        max_matching = max(max_matching, len(mkpts_0))
        filenames[class_img] = len(mkpts_0)
        n_class_imgs += 1
    
    z = time.time()
    print("Each query image takes: ", z - y)
    n_query_imgs += 1
    filenames = {k: v for k, v in sorted(filenames.items(), reverse=True, key=lambda item: item[1])}
    
    for index, (reference_img_name, y) in enumerate(filenames.items()):
        if index >= threshold:
            break
        
        ref_img_path = os.path.join(class_imgs_dir, reference_img_name)
        que_img_path = os.path.join(query_imgs_dir, query_img)
        output_dir = os.path.join(output_dir_path, query_img)
        print(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        shutil.copyfile(que_img_path, os.path.join(output_dir, "query_img_" + query_img))
        shutil.copyfile(ref_img_path, os.path.join(output_dir, reference_img_name + "_" + str(y)))

b = time.time()
print("Total seconds: ", b - a)
print("Query images: ", n_query_imgs)
print("Class images: ", n_class_imgs)
print("Total comparisons: ", n_query_imgs * n_class_imgs)
