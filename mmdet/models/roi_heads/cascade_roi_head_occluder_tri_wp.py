from turtle import position
import torch
import torch.nn as nn

from mmdet.core import (bbox2result, bbox2roi, bbox_mapping, build_assigner,
                        build_sampler, merge_aug_bboxes, merge_aug_masks,
                        multiclass_nms)
from ..builder import HEADS, build_head, build_roi_extractor
from .base_roi_head_occluder_tri import TriOccluderBaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin, TriCat768ConnectedOccluderMaskTestMixin

import ipdb
import cv2
import numpy as np
import torch.nn.functional as F



@HEADS.register_module()
class WPTriOccluderCascadeRoIHead(TriOccluderBaseRoIHead, BBoxTestMixin, TriCat768ConnectedOccluderMaskTestMixin):
    """Cascade roi head including one bbox head and one mask head.

    https://arxiv.org/abs/1712.00726
    """

    def __init__(self,
                 num_stages,
                 stage_loss_weights,
                 enlarge_rpn_box_factor=-1,
                 forward_bbox_2_mask=False,
                 enlarge_proposal_box=True,
                 cascade_return_test_occ_mask=False,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 front_mask_roi_extractor=None,
                 front_mask_head=None,
                 back_mask_roi_extractor=None,
                 back_mask_head=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None):
        assert bbox_roi_extractor is not None
        assert bbox_head is not None
        assert shared_head is None, \
            'Shared head is not supported in Cascade RCNN anymore'
        self.num_stages = num_stages
        self.stage_loss_weights = stage_loss_weights
        self.enlarge_rpn_box_factor = enlarge_rpn_box_factor
        self.forward_bbox_2_mask = forward_bbox_2_mask
        self.enlarge_proposal_box = enlarge_proposal_box
        self.cascade_return_test_occ_mask = cascade_return_test_occ_mask
        # self.
        super(WPTriOccluderCascadeRoIHead, self).__init__(
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            mask_roi_extractor=mask_roi_extractor,
            mask_head=mask_head,
            front_mask_roi_extractor=front_mask_roi_extractor,
            front_mask_head=front_mask_head,
            back_mask_roi_extractor=back_mask_roi_extractor,
            back_mask_head=back_mask_head,
            shared_head=shared_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            # return_test_occ_mask = return_test_occ_mask,
            enlarge_rpn_box_factor = enlarge_rpn_box_factor)

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize box head and box roi extractor.

        Args:
            bbox_roi_extractor (dict): Config of box roi extractor.
            bbox_head (dict): Config of box in box head.
        """
        self.bbox_roi_extractor = nn.ModuleList()
        self.bbox_head = nn.ModuleList()
        if not isinstance(bbox_roi_extractor, list):
            bbox_roi_extractor = [
                bbox_roi_extractor for _ in range(self.num_stages)
            ]
        if not isinstance(bbox_head, list):
            bbox_head = [bbox_head for _ in range(self.num_stages)]
        assert len(bbox_roi_extractor) == len(bbox_head) == self.num_stages
        for roi_extractor, head in zip(bbox_roi_extractor, bbox_head):
            self.bbox_roi_extractor.append(build_roi_extractor(roi_extractor))
            self.bbox_head.append(build_head(head))

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize mask head and mask roi extractor.

        Args:
            mask_roi_extractor (dict): Config of mask roi extractor.
            mask_head (dict): Config of mask in mask head.
        """
        self.mask_head = nn.ModuleList()
        if not isinstance(mask_head, list):
            mask_head = [mask_head for _ in range(self.num_stages)]
        assert len(mask_head) == self.num_stages
        for head in mask_head:
            self.mask_head.append(build_head(head))
        if mask_roi_extractor is not None:
            self.share_roi_extractor = False
            self.mask_roi_extractor = nn.ModuleList()
            if not isinstance(mask_roi_extractor, list):
                mask_roi_extractor = [
                    mask_roi_extractor for _ in range(self.num_stages)
                ]
            assert len(mask_roi_extractor) == self.num_stages
            for roi_extractor in mask_roi_extractor:
                self.mask_roi_extractor.append(
                    build_roi_extractor(roi_extractor))
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor

    def init_front_mask_head(self, front_mask_roi_extractor, front_mask_head):
        """Initialize front mask head and front mask roi extractor.

        Args:
            front_mask_roi_extractor (dict): Config of mask roi extractor.
            front_mask_head (dict): Config of mask in mask head.
        """
        self.front_mask_head = nn.ModuleList()
        if not isinstance(front_mask_head, list):
            front_mask_head = [front_mask_head for _ in range(self.num_stages)]
        assert len(front_mask_head) == self.num_stages
        for head in front_mask_head:
            self.front_mask_head.append(build_head(head))
        if front_mask_roi_extractor is not None:
            self.share_roi_extractor = False
            self.front_mask_roi_extractor = nn.ModuleList()
            if not isinstance(front_mask_roi_extractor, list):
                front_mask_roi_extractor = [
                    front_mask_roi_extractor for _ in range(self.num_stages)
                ]
            assert len(front_mask_roi_extractor) == self.num_stages
            for roi_extractor in front_mask_roi_extractor:
                self.front_mask_roi_extractor.append(
                    build_roi_extractor(roi_extractor))
        else:
            self.share_roi_extractor = True
            self.front_mask_roi_extractor = self.bbox_roi_extractor

    def init_back_mask_head(self, back_mask_roi_extractor, back_mask_head):
        """Initialize front mask head and front mask roi extractor.

        Args:
            back_mask_roi_extractor (dict): Config of mask roi extractor.
            back_mask_head (dict): Config of mask in mask head.
        """
        self.back_mask_head = nn.ModuleList()
        if not isinstance(back_mask_head, list):
             back_mask_head = [back_mask_head for _ in range(self.num_stages)]
        assert len(back_mask_head) == self.num_stages
        for head in back_mask_head:
            self.back_mask_head.append(build_head(head))
        if back_mask_roi_extractor is not None:
            self.share_roi_extractor = False
            self.back_mask_roi_extractor = nn.ModuleList()
            if not isinstance(back_mask_roi_extractor, list):
                back_mask_roi_extractor = [
                    back_mask_roi_extractor for _ in range(self.num_stages)
                ]
            assert len(back_mask_roi_extractor) == self.num_stages
            for roi_extractor in back_mask_roi_extractor:
                self.back_mask_roi_extractor.append(
                    build_roi_extractor(roi_extractor))
        else:
            self.share_roi_extractor = True
            self.back_mask_roi_extractor = self.bbox_roi_extractor
    
    def init_assigner_sampler(self):
        """Initialize assigner and sampler for each stage."""
        self.bbox_assigner = []
        self.bbox_sampler = []
        if self.train_cfg is not None:
            for idx, rcnn_train_cfg in enumerate(self.train_cfg):
                self.bbox_assigner.append(
                    build_assigner(rcnn_train_cfg.assigner))
                self.current_stage = idx
                self.bbox_sampler.append(
                    build_sampler(rcnn_train_cfg.sampler, context=self))

    def init_weights(self, pretrained):
        """Initialize the weights in head.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        for i in range(self.num_stages):
            if self.with_bbox:
                self.bbox_roi_extractor[i].init_weights()
                self.bbox_head[i].init_weights()
            if self.with_mask:
                if not self.share_roi_extractor:
                    self.mask_roi_extractor[i].init_weights()
                self.mask_head[i].init_weights()
            if self.with_front_mask:
                if not self.share_roi_extractor:
                    self.front_mask_roi_extractor[i].init_weights()
                self.front_mask_head[i].init_weights()
            if self.with_back_mask:
                if not self.share_roi_extractor:
                    self.back_mask_roi_extractor[i].init_weights()
                self.back_mask_head[i].init_weights()


    def _bbox_forward_weighted_pooling(self, stage, x, rois, mask_pred):
        # suggested by weidi
        mask_pred = mask_pred + torch.ones(mask_pred.shape).to(mask_pred.device) * 1e-45
        """Box head forward function used in both training and testing."""
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                        rois)
        
        # added by guanqi: for weighted pooling (RoI feature re-weighting)
        mask_pred = mask_pred.repeat(1, bbox_feats.shape[1], 1, 1)
        bbox_feats = bbox_feats * mask_pred
        bbox_feats = F.interpolate(bbox_feats, 7, mode='bilinear')

        # do not support caffe_c4 model anymore
        cls_score, bbox_pred = bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results
    
    def _bbox_forward_vanilla(self, stage, x, rois):
        """Box head forward function used in both training and testing."""
        bbox_roi_extractor = self.bbox_roi_extractor[0]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                        rois)
        # do not support caffe_c4 model anymore
        cls_score, bbox_pred = bbox_head(bbox_feats)
        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results
    
    def _bbox_forward(self, stage, x, rois):
        """Box head forward function used in both training and testing."""
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                        rois)
        # do not support caffe_c4 model anymore
        cls_score, bbox_pred = bbox_head(bbox_feats)
        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def _bbox_forward_train(self, stage, x, sampling_results, gt_bboxes,
                            gt_labels, rcnn_train_cfg, mask_pred=None, label_list=None):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        if mask_pred is None:
            # normal bbox_forward
            bbox_results = self._bbox_forward(stage, x, rois)
        else:
            # weighted RoI pooling (RoI feature re-weighting)
            pred_labels = torch.tensor([]).to(mask_pred.device)
            for iter_i in range(len(sampling_results)):
                pos_inds = sampling_results[iter_i].pos_inds
                pos_labels = label_list[iter_i][pos_inds]
                neg_inds = sampling_results[iter_i].neg_inds
                neg_labels = label_list[iter_i][neg_inds]
                pred_labels = torch.cat([pred_labels, pos_labels, neg_labels], dim=0)
            inds = torch.stack((pred_labels,), 1).unsqueeze(2).unsqueeze(3).repeat(1, 1, mask_pred.shape[2], mask_pred.shape[3])
            forward_mask_pred = torch.gather(mask_pred, 1, inds.to(int))
            forward_mask_pred = torch.sigmoid(forward_mask_pred)
            bbox_results = self._bbox_forward_weighted_pooling(stage, x, rois, forward_mask_pred)
        bbox_targets = self.bbox_head[stage].get_targets(
            sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg)
        loss_bbox = self.bbox_head[stage].loss(bbox_results['cls_score'],
                                               bbox_results['bbox_pred'], rois,
                                               *bbox_targets)

        bbox_results.update(
            loss_bbox=loss_bbox, rois=rois, bbox_targets=bbox_targets)
        return bbox_results

    def _mask_forward(self, stage, x, rois, middle_feat):
        """Mask head forward function used in both training and testing."""
        mask_roi_extractor = self.mask_roi_extractor[stage]
        mask_head = self.mask_head[stage]
        mask_feats = mask_roi_extractor(x[:mask_roi_extractor.num_inputs],
                                        rois)
        # do not support caffe_c4 model anymore
        mask_pred = mask_head(torch.cat([mask_feats, middle_feat], dim=1))

        mask_results = dict(mask_pred=mask_pred)
        return mask_results

    def _mask_forward_train(self,
                            stage,
                            x,
                            sampling_results,
                            middle_feat,
                            gt_masks,
                            rcnn_train_cfg,
                            bbox_feats=None,
                            also_detect_neg=False):
        """Run forward function and calculate loss for mask head in
        training."""
        if also_detect_neg == True:
            # for RoI feature re-weighting
            pos_rois = bbox2roi([res.bboxes for res in sampling_results])
        else:
            # normal forward
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
        mask_results = self._mask_forward(stage, x, pos_rois, middle_feat)
        if also_detect_neg == True:
            return mask_results

        mask_targets = self.mask_head[stage].get_targets(
            sampling_results, gt_masks, rcnn_train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head[stage].loss(mask_results['mask_pred'],
                                               mask_targets, pos_labels)

        mask_results.update(loss_mask=loss_mask)
        return mask_results

    def _front_mask_forward(self, stage, x, rois):
        """Mask head forward function used in both training and testing."""
        mask_roi_extractor = self.front_mask_roi_extractor[stage]
        mask_head = self.front_mask_head[stage]
        mask_feats = mask_roi_extractor(x[:mask_roi_extractor.num_inputs],
                                        rois)
        # do not support caffe_c4 model anymore
        middle_feat, mask_pred = mask_head(mask_feats)

        mask_results = dict(mask_pred=mask_pred, middle_feat=middle_feat)
        return mask_results

    def _front_mask_forward_train(self,
                            stage,
                            x,
                            sampling_results,
                            gt_masks,
                            rcnn_train_cfg,
                            bbox_feats=None,
                            also_detect_neg=False):
        """Run forward function and calculate loss for mask head in
        training."""
        if also_detect_neg == True:
            pos_rois = bbox2roi([res.bboxes for res in sampling_results])
        else:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
        mask_results = self._front_mask_forward(stage, x, pos_rois)
        if also_detect_neg == True:
            return mask_results

        mask_targets = self.front_mask_head[stage].get_targets(
            sampling_results, gt_masks, rcnn_train_cfg)
        # ipdb.set_trace()
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.front_mask_head[stage].loss(mask_results['mask_pred'],
                                               mask_targets, pos_labels)
        
        mask_results.update(loss_front_mask=loss_mask, mask_targets=mask_targets)
        return mask_results

    def _back_mask_forward(self, stage, x, rois):
        """Mask head forward function used in both training and testing."""
        mask_roi_extractor = self.back_mask_roi_extractor[stage]
        mask_head = self.back_mask_head[stage]
        mask_feats = mask_roi_extractor(x[:mask_roi_extractor.num_inputs],
                                        rois)
        # do not support caffe_c4 model anymore
        middle_feat, mask_pred = mask_head(mask_feats)

        mask_results = dict(mask_pred=mask_pred, middle_feat=middle_feat)
        return mask_results

    def _back_mask_forward_train(self,
                            stage,
                            x,
                            sampling_results,
                            gt_masks,
                            rcnn_train_cfg,
                            bbox_feats=None,
                            also_detect_neg=False):
        """Run forward function and calculate loss for mask head in
        training."""
        if also_detect_neg == True:
            pos_rois = bbox2roi([res.bboxes for res in sampling_results])
        else:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
        mask_results = self._back_mask_forward(stage, x, pos_rois)
        if also_detect_neg == True:
            return mask_results

        mask_targets = self.back_mask_head[stage].get_targets(
            sampling_results, gt_masks, rcnn_train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.back_mask_head[stage].loss(mask_results['mask_pred'],
                                               mask_targets, pos_labels)
        
        mask_results.update(loss_back_mask=loss_mask, mask_targets=mask_targets)
        return mask_results
    
    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_front_masks=None,
                      gt_back_masks=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        losses = dict()
        for i in range(self.num_stages):
            self.current_stage = i
            rcnn_train_cfg = self.train_cfg[i]
            lw = self.stage_loss_weights[i]

            # assign gts and sample proposals
            sampling_results = []
            if self.with_bbox or self.with_mask:
                bbox_assigner = self.bbox_assigner[i]
                bbox_sampler = self.bbox_sampler[i]
                num_imgs = len(img_metas)
                if gt_bboxes_ignore is None:
                    gt_bboxes_ignore = [None for _ in range(num_imgs)]

                for j in range(num_imgs):
                    if i == 1:
                        # new label for RoI feature re-weighting
                        label_list[j] = torch.cat([gt_labels[j], label_list[j]])
                    assign_result = bbox_assigner.assign(
                        proposal_list[j], gt_bboxes[j], gt_bboxes_ignore[j],
                        gt_labels[j])
                    sampling_result = bbox_sampler.sample(
                        assign_result,
                        proposal_list[j],
                        gt_bboxes[j],
                        gt_labels[j],
                        feats=[lvl_feat[j][None] for lvl_feat in x])
                    sampling_results.append(sampling_result)         
                    
            # bbox head forward and loss
            if i == 0:
                # normal forward
                bbox_results = self._bbox_forward_train(i, x, sampling_results,
                                                        gt_bboxes, gt_labels,
                                                        rcnn_train_cfg)
            else:
                # RoI feature re-weighting: get the predictions of the i-1 th mask heads in terms of the given boxes
                front_mask_results = self._front_mask_forward_train(
                    i-1, x, sampling_results, gt_front_masks, rcnn_train_cfg,
                    bbox_results['bbox_feats'], also_detect_neg=True)
                back_mask_results = self._back_mask_forward_train(
                    i-1, x, sampling_results, gt_back_masks, rcnn_train_cfg,
                    bbox_results['bbox_feats'], also_detect_neg=True)
                mask_results = self._mask_forward_train(
                    i-1, x, sampling_results, torch.cat([front_mask_results['middle_feat'], back_mask_results['middle_feat']], dim=1), gt_masks, rcnn_train_cfg,
                    bbox_results['bbox_feats'], also_detect_neg=True)
                bbox_results = self._bbox_forward_train(i, x, sampling_results,
                                                        gt_bboxes, gt_labels,
                                                        rcnn_train_cfg, mask_results['mask_pred'], label_list)


            for name, value in bbox_results['loss_bbox'].items():
                losses[f's{i}.{name}'] = (
                    value * lw if 'loss' in name else value)

            if self.forward_bbox_2_mask:
                # forward box predictions to mask heads
                # refine bboxes
                if i < self.num_stages:
                    pos_is_gts = [res.pos_is_gt for res in sampling_results]
                    # bbox_targets is a tuple
                    roi_labels = bbox_results['bbox_targets'][0]
                    with torch.no_grad():
                        roi_labels = torch.where(
                            roi_labels == self.bbox_head[i].num_classes,
                            bbox_results['cls_score'][:, :-1].argmax(1),
                            roi_labels)
                        if i == 0:
                            proposal_list, label_list = self.bbox_head[i].refine_bboxes_wp(
                                bbox_results['rois'], roi_labels,
                                bbox_results['bbox_pred'], pos_is_gts, img_metas)
                        else:
                            proposal_list = self.bbox_head[i].refine_bboxes(
                                bbox_results['rois'], roi_labels,
                                bbox_results['bbox_pred'], pos_is_gts, img_metas)
                        # ipdb.set_trace()
                # new sampling results
                sampling_results = []
                for j in range(num_imgs):
                    assign_result = bbox_assigner.assign(
                        proposal_list[j], gt_bboxes[j], gt_bboxes_ignore[j],
                        gt_labels[j])
                    sampling_result = bbox_sampler.sample(
                        assign_result,
                        proposal_list[j],
                        gt_bboxes[j],
                        gt_labels[j],
                        feats=[lvl_feat[j][None] for lvl_feat in x])
                    sampling_results.append(sampling_result)
                    # ipdb.set_trace()
                
            # front mask head forward and loss
            if self.with_front_mask:
                # ipdb.set_trace()
                front_mask_results = self._front_mask_forward_train(
                    i, x, sampling_results, gt_front_masks, rcnn_train_cfg,
                    bbox_results['bbox_feats'])
                losses[f's{i}.loss_front_mask'] = front_mask_results['loss_front_mask']['loss_mask'] * lw

            # back mask head forward and loss
            if self.with_back_mask:
                back_mask_results = self._back_mask_forward_train(
                    i, x, sampling_results, gt_back_masks, rcnn_train_cfg,
                    bbox_results['bbox_feats'])
                losses[f's{i}.loss_back_mask'] = back_mask_results['loss_back_mask']['loss_mask'] * lw
            
            # mask head forward and loss
            if self.with_mask:
                mask_results = self._mask_forward_train(
                    i, x, sampling_results, torch.cat([front_mask_results['middle_feat'], back_mask_results['middle_feat']], dim=1), gt_masks, rcnn_train_cfg,
                    bbox_results['bbox_feats'])
                for name, value in mask_results['loss_mask'].items():
                    losses[f's{i}.{name}'] = (
                        value * lw if 'loss' in name else value)


            if not self.forward_bbox_2_mask:
                # otherwise, update the boxes at the end of iteration
                # refine bboxes
                if i < self.num_stages - 1:
                    pos_is_gts = [res.pos_is_gt for res in sampling_results]
                    # bbox_targets is a tuple
                    roi_labels = bbox_results['bbox_targets'][0]
                    with torch.no_grad():
                        roi_labels = torch.where(
                            roi_labels == self.bbox_head[i].num_classes,
                            bbox_results['cls_score'][:, :-1].argmax(1),
                            roi_labels)
                        proposal_list = self.bbox_head[i].refine_bboxes(
                            bbox_results['rois'], roi_labels,
                            bbox_results['bbox_pred'], pos_is_gts, img_metas)
        
        return losses

    
    # for weighted pooling inference
    def simple_test(self, x, proposal_list, img_metas, rescale=False):
    # def simple_test_normal_wp_inference(self, x, proposal_list, img_metas, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        num_imgs = len(proposal_list)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # "ms" in variable names means multi-stage
        ms_bbox_result = {}
        ms_segm_result = {}
        ms_scores = []
        rcnn_test_cfg = self.test_cfg

        rois = bbox2roi(proposal_list)

        
        # the first iteration
        for i in [0]:

            bbox_results = self._bbox_forward(i, x, rois)

            # split batch bbox prediction back to each image
            cls_score = bbox_results['cls_score']
            bbox_pred = bbox_results['bbox_pred']
            num_proposals_per_img = tuple(
                len(proposals) for proposals in proposal_list)
            rois = rois.split(num_proposals_per_img, 0)
            cls_score = cls_score.split(num_proposals_per_img, 0)
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head[i].bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
            # ms_scores.append(cls_score)

            if i < 1:
                bbox_label = [s[:, :-1].argmax(dim=1) for s in cls_score]
                # ipdb.set_trace()
                rois = torch.cat([
                    self.bbox_head[i].regress_by_class(rois[j], bbox_label[j],
                                                       bbox_pred[j],
                                                       img_metas[j])
                    for j in range(num_imgs)
                ])

        # get mask prediction for each b1
        front_mask_results = self._front_mask_forward(i, x, rois)
        back_mask_results = self._back_mask_forward(i, x, rois)
        # torch.cuda.empty_cache()
        mask_results = self._mask_forward(i, x, rois, torch.cat([front_mask_results['middle_feat'], back_mask_results['middle_feat']], dim=1))
        mask_pred = mask_results['mask_pred']
        label = bbox_label[0]
        inds = torch.stack((label,), 1).unsqueeze(2).unsqueeze(3).repeat(1, 1, mask_pred.shape[2], mask_pred.shape[3])
        # ipdb.set_trace()
        mask_pred = torch.gather(mask_pred, 1, inds)
        

        # the second iteration
        for i in [1]:

            bbox_results = self._bbox_forward_weighted_pooling(i, x, rois, torch.sigmoid(mask_pred))

            # split batch bbox prediction back to each image
            cls_score = bbox_results['cls_score']
            bbox_pred = bbox_results['bbox_pred']
            num_proposals_per_img = tuple(
                len(proposals) for proposals in proposal_list)
            rois = rois.split(num_proposals_per_img, 0)
            cls_score = cls_score.split(num_proposals_per_img, 0)
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head[i].bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
            ms_scores.append(cls_score)

        
        # average scores of each image by stages, only for the output of the 2nd iteration
        cls_score = [
            sum([score[i] for score in ms_scores]) / float(len(ms_scores))
            for i in range(num_imgs)
        ]
        
        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(num_imgs):
            det_bbox, det_label = self.bbox_head[-1].get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)

        # ipdb.set_trace()
        if torch.onnx.is_in_onnx_export():
            return det_bboxes, det_labels
        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head[-1].num_classes)
            for i in range(num_imgs)
        ]
        ms_bbox_result['ensemble'] = bbox_results

        if self.with_mask:
            if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
                mask_classes = self.mask_head[-1].num_classes
                segm_results = [[[] for _ in range(mask_classes)]
                                for _ in range(num_imgs)]
            else:
                if rescale and not isinstance(scale_factors[0], float):
                    scale_factors = [
                        torch.from_numpy(scale_factor).to(det_bboxes[0].device)
                        for scale_factor in scale_factors
                    ]
                _bboxes = [
                    det_bboxes[i][:, :4] *
                    scale_factors[i] if rescale else det_bboxes[i][:, :4]
                    for i in range(len(det_bboxes))
                ]
                mask_rois = bbox2roi(_bboxes)
                num_mask_rois_per_img = tuple(
                    _bbox.size(0) for _bbox in _bboxes)
                aug_masks = []
                if self.cascade_return_test_occ_mask == True:
                    front_aug_masks = []
                    back_aug_masks = []
                for i in range(self.num_stages):
                    front_mask_results = self._front_mask_forward(i, x, mask_rois)
                    back_mask_results = self._back_mask_forward(i, x, mask_rois)
                    mask_results = self._mask_forward(i, x, mask_rois, torch.cat([front_mask_results['middle_feat'], back_mask_results['middle_feat']], dim=1))

                    mask_pred = mask_results['mask_pred']
                    # split batch mask prediction back to each image
                    mask_pred = mask_pred.split(num_mask_rois_per_img, 0)
                    aug_masks.append(
                        [m.sigmoid().cpu().detach().numpy() for m in mask_pred])
                    if self.cascade_return_test_occ_mask == True:
                        # ipdb.set_trace()
                        front_mask_pred = front_mask_results['mask_pred']
                        # split batch mask prediction back to each image
                        front_mask_pred = front_mask_pred.split(num_mask_rois_per_img, 0)
                        front_aug_masks.append(
                            [m.sigmoid().cpu().numpy() for m in front_mask_pred])
                        back_mask_pred = back_mask_results['mask_pred']
                        # split batch mask prediction back to each image
                        back_mask_pred = back_mask_pred.split(num_mask_rois_per_img, 0)
                        back_aug_masks.append(
                            [m.sigmoid().cpu().numpy() for m in back_mask_pred])

                # apply mask post-processing to each image individually
                segm_results = []
                if self.cascade_return_test_occ_mask == True:
                    front_segm_results = []
                    back_segm_results = []
                for i in range(num_imgs):
                    if det_bboxes[i].shape[0] == 0:
                        segm_results.append(
                            [[]
                             for _ in range(self.mask_head[-1].num_classes)])
                    else:
                        aug_mask = [mask[i] for mask in aug_masks]
                        merged_masks = merge_aug_masks(
                            aug_mask, [[img_metas[i]]] * (self.num_stages),
                            rcnn_test_cfg)
                        segm_result = self.mask_head[-1].get_seg_masks(
                            merged_masks, _bboxes[i], det_labels[i],
                            rcnn_test_cfg, ori_shapes[i], scale_factors[i],
                            rescale)
                        segm_results.append(segm_result)
                        if self.cascade_return_test_occ_mask == True:
                            # ipdb.set_trace()
                            front_aug_mask = [mask[i] for mask in front_aug_masks]
                            front_merged_masks = merge_aug_masks(
                                front_aug_mask, [[img_metas[i]]] * (self.num_stages),
                                rcnn_test_cfg)
                            front_segm_result = self.mask_head[-1].get_seg_masks(
                                front_merged_masks, _bboxes[i], det_labels[i],
                                rcnn_test_cfg, ori_shapes[i], scale_factors[i],
                                rescale)
                            front_segm_results.append(front_segm_result)
                            back_aug_mask = [mask[i] for mask in back_aug_masks]
                            back_merged_masks = merge_aug_masks(
                                back_aug_mask, [[img_metas[i]]] * (self.num_stages),
                                rcnn_test_cfg)
                            back_segm_result = self.mask_head[-1].get_seg_masks(
                                back_merged_masks, _bboxes[i], det_labels[i],
                                rcnn_test_cfg, ori_shapes[i], scale_factors[i],
                                rescale)
                            back_segm_results.append(back_segm_result)
            ms_segm_result['ensemble'] = segm_results

        # ipdb.set_trace()
        if self.with_mask:
            # ipdb.set_trace()
            if self.cascade_return_test_occ_mask == True:
                results = list(
                    zip(ms_bbox_result['ensemble'], ms_segm_result['ensemble'], front_segm_results, back_segm_results))
            else:
                results = list(
                    zip(ms_bbox_result['ensemble'], ms_segm_result['ensemble']))
        else:
            results = ms_bbox_result['ensemble']

        return results
    
    
    def simple_test_org(self, x, proposal_list, img_metas, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        num_imgs = len(proposal_list)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # "ms" in variable names means multi-stage
        ms_bbox_result = {}
        ms_segm_result = {}
        ms_scores = []
        rcnn_test_cfg = self.test_cfg

        rois = bbox2roi(proposal_list)

        for i in range(self.num_stages):
            
            bbox_results = self._bbox_forward(i, x, rois)

            # split batch bbox prediction back to each image
            cls_score = bbox_results['cls_score']
            bbox_pred = bbox_results['bbox_pred']
            num_proposals_per_img = tuple(
                len(proposals) for proposals in proposal_list)
            rois = rois.split(num_proposals_per_img, 0)
            cls_score = cls_score.split(num_proposals_per_img, 0)
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head[i].bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
            ms_scores.append(cls_score)

            if i < self.num_stages - 1:
                bbox_label = [s[:, :-1].argmax(dim=1) for s in cls_score]
                rois = torch.cat([
                    self.bbox_head[i].regress_by_class(rois[j], bbox_label[j],
                                                       bbox_pred[j],
                                                       img_metas[j])
                    for j in range(num_imgs)
                ])

        # average scores of each image by stages
        cls_score = [
            sum([score[i] for score in ms_scores]) / float(len(ms_scores))
            for i in range(num_imgs)
        ]
        
        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(num_imgs):
            det_bbox, det_label = self.bbox_head[-1].get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)

        # ipdb.set_trace()
        if torch.onnx.is_in_onnx_export():
            return det_bboxes, det_labels
        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head[-1].num_classes)
            for i in range(num_imgs)
        ]
        ms_bbox_result['ensemble'] = bbox_results

        if self.with_mask:
            if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
                mask_classes = self.mask_head[-1].num_classes
                segm_results = [[[] for _ in range(mask_classes)]
                                for _ in range(num_imgs)]
            else:
                if rescale and not isinstance(scale_factors[0], float):
                    scale_factors = [
                        torch.from_numpy(scale_factor).to(det_bboxes[0].device)
                        for scale_factor in scale_factors
                    ]
                _bboxes = [
                    det_bboxes[i][:, :4] *
                    scale_factors[i] if rescale else det_bboxes[i][:, :4]
                    for i in range(len(det_bboxes))
                ]
                mask_rois = bbox2roi(_bboxes)
                num_mask_rois_per_img = tuple(
                    _bbox.size(0) for _bbox in _bboxes)
                aug_masks = []
                if self.cascade_return_test_occ_mask == True:
                    front_aug_masks = []
                    back_aug_masks = []
                for i in range(self.num_stages):
                    front_mask_results = self._front_mask_forward(i, x, mask_rois)
                    back_mask_results = self._back_mask_forward(i, x, mask_rois)
                    mask_results = self._mask_forward(i, x, mask_rois, torch.cat([front_mask_results['middle_feat'], back_mask_results['middle_feat']], dim=1))

                    mask_pred = mask_results['mask_pred']
                    # split batch mask prediction back to each image
                    mask_pred = mask_pred.split(num_mask_rois_per_img, 0)
                    aug_masks.append(
                        [m.sigmoid().cpu().numpy() for m in mask_pred])
                    if self.cascade_return_test_occ_mask == True:
                        # ipdb.set_trace()
                        front_mask_pred = front_mask_results['mask_pred']
                        # split batch mask prediction back to each image
                        front_mask_pred = front_mask_pred.split(num_mask_rois_per_img, 0)
                        front_aug_masks.append(
                            [m.sigmoid().cpu().numpy() for m in front_mask_pred])
                        back_mask_pred = back_mask_results['mask_pred']
                        # split batch mask prediction back to each image
                        back_mask_pred = back_mask_pred.split(num_mask_rois_per_img, 0)
                        back_aug_masks.append(
                            [m.sigmoid().cpu().numpy() for m in back_mask_pred])

                # apply mask post-processing to each image individually
                segm_results = []
                if self.cascade_return_test_occ_mask == True:
                    front_segm_results = []
                    back_segm_results = []
                for i in range(num_imgs):
                    if det_bboxes[i].shape[0] == 0:
                        segm_results.append(
                            [[]
                             for _ in range(self.mask_head[-1].num_classes)])
                    else:
                        aug_mask = [mask[i] for mask in aug_masks]
                        merged_masks = merge_aug_masks(
                            aug_mask, [[img_metas[i]]] * (self.num_stages),
                            rcnn_test_cfg)
                        segm_result = self.mask_head[-1].get_seg_masks(
                            merged_masks, _bboxes[i], det_labels[i],
                            rcnn_test_cfg, ori_shapes[i], scale_factors[i],
                            rescale)
                        segm_results.append(segm_result)
                        if self.cascade_return_test_occ_mask == True:
                            # ipdb.set_trace()
                            front_aug_mask = [mask[i] for mask in front_aug_masks]
                            front_merged_masks = merge_aug_masks(
                                front_aug_mask, [[img_metas[i]]] * (self.num_stages),
                                rcnn_test_cfg)
                            front_segm_result = self.mask_head[-1].get_seg_masks(
                                front_merged_masks, _bboxes[i], det_labels[i],
                                rcnn_test_cfg, ori_shapes[i], scale_factors[i],
                                rescale)
                            front_segm_results.append(front_segm_result)
                            back_aug_mask = [mask[i] for mask in back_aug_masks]
                            back_merged_masks = merge_aug_masks(
                                back_aug_mask, [[img_metas[i]]] * (self.num_stages),
                                rcnn_test_cfg)
                            back_segm_result = self.mask_head[-1].get_seg_masks(
                                back_merged_masks, _bboxes[i], det_labels[i],
                                rcnn_test_cfg, ori_shapes[i], scale_factors[i],
                                rescale)
                            back_segm_results.append(back_segm_result)
            ms_segm_result['ensemble'] = segm_results

        # ipdb.set_trace()
        if self.with_mask:
            # ipdb.set_trace()
            if self.cascade_return_test_occ_mask == True:
                results = list(
                    zip(ms_bbox_result['ensemble'], ms_segm_result['ensemble'], front_segm_results, back_segm_results))
            else:
                results = list(
                    zip(ms_bbox_result['ensemble'], ms_segm_result['ensemble']))
        else:
            results = ms_bbox_result['ensemble']

        return results

    def aug_test(self, features, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        rcnn_test_cfg = self.test_cfg
        aug_bboxes = []
        aug_scores = []
        for x, img_meta in zip(features, img_metas):
            # only one image in the batch
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']
            flip_direction = img_meta[0]['flip_direction']

            proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
                                     scale_factor, flip, flip_direction)
            # "ms" in variable names means multi-stage
            ms_scores = []

            rois = bbox2roi([proposals])
            for i in range(self.num_stages):
                bbox_results = self._bbox_forward(i, x, rois)
                ms_scores.append(bbox_results['cls_score'])

                if i < self.num_stages - 1:
                    bbox_label = bbox_results['cls_score'][:, :-1].argmax(
                        dim=1)
                    rois = self.bbox_head[i].regress_by_class(
                        rois, bbox_label, bbox_results['bbox_pred'],
                        img_meta[0])

            cls_score = sum(ms_scores) / float(len(ms_scores))
            bboxes, scores = self.bbox_head[-1].get_bboxes(
                rois,
                cls_score,
                bbox_results['bbox_pred'],
                img_shape,
                scale_factor,
                rescale=False,
                cfg=None)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
        det_bboxes, det_labels = multiclass_nms(merged_bboxes, merged_scores,
                                                rcnn_test_cfg.score_thr,
                                                rcnn_test_cfg.nms,
                                                rcnn_test_cfg.max_per_img)

        bbox_result = bbox2result(det_bboxes, det_labels,
                                  self.bbox_head[-1].num_classes)

        if self.with_mask:
            if det_bboxes.shape[0] == 0:
                segm_result = [[[]
                                for _ in range(self.mask_head[-1].num_classes)]
                               ]
            else:
                aug_masks = []
                aug_img_metas = []
                for x, img_meta in zip(features, img_metas):
                    img_shape = img_meta[0]['img_shape']
                    scale_factor = img_meta[0]['scale_factor']
                    flip = img_meta[0]['flip']
                    flip_direction = img_meta[0]['flip_direction']
                    _bboxes = bbox_mapping(det_bboxes[:, :4], img_shape,
                                           scale_factor, flip, flip_direction)
                    mask_rois = bbox2roi([_bboxes])
                    for i in range(self.num_stages):
                        mask_results = self._mask_forward(i, x, mask_rois)
                        aug_masks.append(
                            mask_results['mask_pred'].sigmoid().cpu().numpy())
                        aug_img_metas.append(img_meta)
                merged_masks = merge_aug_masks(aug_masks, aug_img_metas,
                                               self.test_cfg)

                ori_shape = img_metas[0][0]['ori_shape']
                segm_result = self.mask_head[-1].get_seg_masks(
                    merged_masks,
                    det_bboxes,
                    det_labels,
                    rcnn_test_cfg,
                    ori_shape,
                    scale_factor=1.0,
                    rescale=False)
            return [(bbox_result, segm_result)]
        else:
            return [bbox_result]
