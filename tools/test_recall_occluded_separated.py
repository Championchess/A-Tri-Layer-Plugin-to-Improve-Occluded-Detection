'''
evaluate the performance of object detectors under occlusion
calculate the recall of models on Separated COCO and Occluded COCO
decompose the reason of failure:
if the tgt_obj fails to be recalled, then see the prediction mask with highest IOU of it
1. if the highest IOU is above IOU_th, (must be predicted as another class) then cls
2. if the highest IOU is [IOU_BG_th, IOU_th], 
    i. same class: loc
    ii. different class: cls+loc
3. if the highest IOU is below IOU_BG_th, then miss
'''

import numpy as np
import mmcv
import ipdb
import json
import math
from pycocotools import mask as coco_mask


def compute_iou_mask(mask1, mask2):
    mask1_area = np.count_nonzero(mask1 == 1)
    mask2_area = np.count_nonzero(mask2 == 1)
    intersection = np.count_nonzero(np.logical_and(mask1==1,  mask2==1))
    iou = intersection/(mask1_area+mask2_area-intersection)
    return iou


# COCO Class ID from 0 to 79
coco_noun_class_list = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush']

def sort_swin_det_result(swin_det_file_pth):
    swin_det_file = mmcv.load(swin_det_file_pth)
    dict_det = {}
    for i in range(len(swin_det_file)):
        print(i)
        cur_img_name = swin_det_file[i][0]
        if cur_img_name not in dict_det.keys():
            dict_det[cur_img_name] = []
        for j in range(len(swin_det_file[i][2])): # for j in 80
            assert len(swin_det_file[i][2][j]) == len(swin_det_file[i][1][j])
            for k in range(len(swin_det_file[i][2][j])):
                cur_binary_mask = coco_mask.decode(swin_det_file[i][2][j][k])
                cur_det_bbox = swin_det_file[i][1][j][k][:4]
                dict_det[cur_img_name].append([swin_det_file[i][1][j][k][4], coco_noun_class_list[j], cur_binary_mask, cur_det_bbox])
        dict_det[cur_img_name].sort(key=lambda x: (-x[0], x[3][0], x[3][1])) # rank by confidence from high to low, avoid same confidence
    print("successfully cope with detections")

    return dict_det


def main(dict_det, result_save_pth, gt_json, is_occ):
        
    CONFIDENCE_THRESHOLD = 0.3
    IOU_THRESHOLD = 0.75
    mask_fail_reason_dict = {"suc":0}
    print(len(gt_json))

    for iter_i in range(len(gt_json)):
        print(iter_i)
        print(mask_fail_reason_dict)
        cur_item = gt_json[iter_i]
        cur_img_name = cur_item[0]
        cur_gt_bbox = cur_item[3]
        cur_anns_id = cur_item[2]
        if is_occ:
            cur_gt_bbox = [cur_gt_bbox[0], cur_gt_bbox[1], cur_gt_bbox[0] + cur_gt_bbox[2], cur_gt_bbox[1] + cur_gt_bbox[3]]
        cur_gt_class = cur_item[1]
        cur_gt_mask = coco_mask.decode(cur_item[4])

        assert cur_img_name in dict_det.keys()
        cur_detections = dict_det[cur_img_name]

        correct_flag = False
        for i in range(len(cur_detections)):
            cur_det_confidence = cur_detections[i][0]
            if cur_det_confidence < CONFIDENCE_THRESHOLD:
                break
            cur_det_class = cur_detections[i][1]
            if cur_det_class != cur_gt_class:
                continue
            cur_det_mask = cur_detections[i][2]
            cur_iou = compute_iou_mask(cur_det_mask, cur_gt_mask)
            if cur_iou >= IOU_THRESHOLD:
                correct_flag = True
                break
        if correct_flag == True:
            mask_fail_reason_dict['suc'] += 1
        
    save_dict_collect = [mask_fail_reason_dict]
    # save the recall results
    with open(result_save_pth, "w") as dump_f:
        json.dump(save_dict_collect, dump_f)
    

if __name__ == '__main__':

    '''
    for swin-t + our plugin
    '''
    swin_det_file_pth = "xxx/swin-t_our_plugin.pkl" # path to save detection results in test.py
    dict_det = sort_swin_det_result(swin_det_file_pth) # sort the detection results by confidence first
    # for occluded coco
    result_save_pth = "xxx/failure_reason_dicts_occ/swin-t_our_plugin.json" # path to save partially occluded recall results
    gt_json = mmcv.load("../data/occluded_coco.pkl") # occluded coco dataset
    main(dict_det, result_save_pth, gt_json, is_occ=True)
    # for separated coco
    result_save_pth = "xxx/failure_reason_dicts_sep/swin-t_our_plugin.json" # path to save separated recall results
    gt_json = mmcv.load("../data/separated_coco.pkl") # separated coco dataset
    main(dict_det, result_save_pth, gt_json, is_occ=False)
    del dict_det
