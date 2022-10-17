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

def compute_iou_bbox(rec1, rec2):
    left_column_max = max(rec1[0], rec2[0])
    right_column_min = min(rec1[2], rec2[2])
    up_row_max = max(rec1[1], rec2[1])
    down_row_min = min(rec1[3], rec2[3])
    # no intersections for two rectangles
    if left_column_max >= right_column_min or down_row_min <= up_row_max:
        return 0
    # intersection exists for two rectangles
    else:
        S1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
        S2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
        S_cross = (down_row_min - up_row_max) * (right_column_min - left_column_max)
        return S_cross / (S1 + S2 - S_cross)

def occlu_ratio_2_level(occlu_ratio):
    occlu_ratio = max(1 - occlu_ratio, 0) # obtain the ratio of occluded parts
    if occlu_ratio == 0:
        return 0
    assert 1 <= math.ceil(occlu_ratio * 10.0) <= 10
    return math.ceil(occlu_ratio * 10.0)


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
    IOU_BG_THRESHOLD = 0.1
    bbox_fail_reason_dict = {"suc":0, "cls":0, "loc":0, "cls+loc":0, "miss":0}
    mask_fail_reason_dict = {"suc":0, "cls":0, "loc(bbox)":0, "loc(mask)":0, "cls+loc":0, "miss":0}
    print(len(gt_json))

    for iter_i in range(len(gt_json)):
        print(iter_i)
        print(bbox_fail_reason_dict)
        print(mask_fail_reason_dict)
        cur_item = gt_json[iter_i]
        cur_img_name = cur_item[0]
        cur_gt_bbox = cur_item[3]
        cur_anns_id = cur_item[2]
        if is_occ:
            cur_gt_bbox = [cur_gt_bbox[0], cur_gt_bbox[1], cur_gt_bbox[0] + cur_gt_bbox[2], cur_gt_bbox[1] + cur_gt_bbox[3]]
        cur_gt_class = cur_item[1]
        cur_gt_mask = coco_mask.decode(cur_item[4])
        cur_occlu_ratio = cur_item[5]
        if cur_occlu_ratio == "NAN":
            continue
        cur_occlu_level = occlu_ratio_2_level(cur_occlu_ratio)

        flag_collect = [] # [mask_flag, bbox_flag]
        assert cur_img_name in dict_det.keys()
        cur_detections = dict_det[cur_img_name]

        # first see whether mask is correctly recalled
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
        flag_collect.append(correct_flag)
        if correct_flag == True:
            vis_mask = cur_det_mask
            vis_class = cur_det_class
            vis_iou = cur_iou
            vis_bbox_iou = compute_iou_bbox(cur_detections[i][3], cur_gt_bbox)
            vis_conf = cur_det_confidence
            vis_bbox = cur_detections[i][3] # visualize the corresponding bbox of the mask
            mask_fail_reason_dict['suc'] += 1
        # second see whether box is correctly recalled
        correct_flag = False
        for i in range(len(cur_detections)):
            cur_det_confidence = cur_detections[i][0]
            if cur_det_confidence < CONFIDENCE_THRESHOLD:
                break
            cur_det_class = cur_detections[i][1]
            if cur_det_class != cur_gt_class:
                continue
            cur_det_bbox = cur_detections[i][3]
            # ipdb.set_trace()
            cur_iou = compute_iou_bbox(cur_det_bbox, cur_gt_bbox)
            if cur_iou >= IOU_THRESHOLD:
                correct_flag = True
                break
        flag_collect.append(correct_flag)
        if correct_flag == True:
            bbox_fail_reason_dict['suc'] += 1
        
        
        # figure out the failure reason
        # first figure out the mask failure reason
        if flag_collect[0] == False:
            max_iou = IOU_BG_THRESHOLD
            max_iou_id = -1 
            for i in range(len(cur_detections)):
                cur_det_confidence = cur_detections[i][0]
                if cur_det_confidence < CONFIDENCE_THRESHOLD:
                    break
                cur_det_class = cur_detections[i][1]
                cur_det_mask = cur_detections[i][2]
                cur_iou = compute_iou_mask(cur_det_mask, cur_gt_mask)
                if cur_iou > max_iou:
                    max_iou = cur_iou
                    max_iou_id = i
            if max_iou_id == -1:
                mask_fail_reason_dict['miss'] += 1
                vis_mask = -1
                vis_class = -1
                vis_conf = -1
                vis_iou = -1
                vis_bbox_iou = -1
                vis_bbox = -1
            else:
                max_iou_conf = cur_detections[max_iou_id][0]
                max_iou_class = cur_detections[max_iou_id][1]
                max_iou_mask = cur_detections[max_iou_id][2]
                max_iou_bbox = cur_detections[max_iou_id][3]
                vis_mask = max_iou_mask
                vis_bbox = max_iou_bbox
                vis_class = max_iou_class
                vis_conf = max_iou_conf
                vis_iou = max_iou
                vis_bbox_iou = compute_iou_bbox(vis_bbox, cur_gt_bbox)
                assert max_iou_conf >= CONFIDENCE_THRESHOLD
                if max_iou >= IOU_THRESHOLD:
                    assert max_iou_class != cur_gt_class
                    mask_fail_reason_dict['cls'] += 1
                else:
                    if max_iou_class == cur_gt_class:
                        if vis_bbox_iou >= IOU_THRESHOLD:
                            mask_fail_reason_dict['loc(mask)'] += 1
                        else:
                            mask_fail_reason_dict['loc(bbox)'] += 1
                    else:
                        mask_fail_reason_dict['cls+loc'] += 1
        # second figure out the box failure reason
        if flag_collect[1] == False:
            max_iou = IOU_BG_THRESHOLD
            max_iou_id = -1 
            for i in range(len(cur_detections)):
                cur_det_confidence = cur_detections[i][0]
                if cur_det_confidence < CONFIDENCE_THRESHOLD:
                    break
                cur_det_class = cur_detections[i][1]
                cur_det_bbox = cur_detections[i][3]
                cur_iou = compute_iou_bbox(cur_det_bbox, cur_gt_bbox)
                if cur_iou > max_iou:
                    max_iou = cur_iou
                    max_iou_id = i
            if max_iou_id == -1:
                bbox_fail_reason_dict['miss'] += 1
            else:
                max_iou_conf = cur_detections[max_iou_id][0]
                max_iou_class = cur_detections[max_iou_id][1]
                max_iou_mask = cur_detections[max_iou_id][2]
                max_iou_bbox = cur_detections[max_iou_id][3]
                assert max_iou_conf >= CONFIDENCE_THRESHOLD
                if max_iou >= IOU_THRESHOLD:
                    assert max_iou_class != cur_gt_class
                    bbox_fail_reason_dict['cls'] += 1
                else:
                    if max_iou_class == cur_gt_class:
                        bbox_fail_reason_dict['loc'] += 1
                    else:
                        bbox_fail_reason_dict['cls+loc'] += 1
                    

    save_dict_collect = [bbox_fail_reason_dict, mask_fail_reason_dict]
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