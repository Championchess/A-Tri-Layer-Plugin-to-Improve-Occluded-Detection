from abc import ABCMeta, abstractmethod

import torch.nn as nn

from ..builder import build_shared_head


class TriOccluderBaseRoIHead(nn.Module, metaclass=ABCMeta):
    """Base class for RoIHeads."""

    def __init__(self,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 front_mask_roi_extractor=None,
                 front_mask_head=None,
                 back_mask_roi_extractor=None,
                 back_mask_head=None,
                 shared_head=None,
                 ignore_empty_front_mask=False,
                 ignore_empty_back_mask=False,
                 train_cfg=None,
                 test_cfg=None,
                 return_test_occ_mask=False,
                 enlarge_rpn_box_factor=-1,
                 test_with_gt_bbox=False):
        super(TriOccluderBaseRoIHead, self).__init__()
        # print("return_test_occ_mask", return_test_occ_mask)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.ignore_empty_front_mask = ignore_empty_front_mask
        self.ignore_empty_back_mask = ignore_empty_back_mask
        self.test_with_gt_bbox = test_with_gt_bbox
        self.enlarge_rpn_box_factor = enlarge_rpn_box_factor
        if shared_head is not None:
            self.shared_head = build_shared_head(shared_head)

        if bbox_head is not None:
            self.init_bbox_head(bbox_roi_extractor, bbox_head)

        if mask_head is not None:
            self.init_mask_head(mask_roi_extractor, mask_head)

        if front_mask_head is not None:
            self.init_front_mask_head(front_mask_roi_extractor, front_mask_head)

        if back_mask_head is not None:
            self.init_back_mask_head(back_mask_roi_extractor, back_mask_head)

        self.init_assigner_sampler()
        self.return_test_occ_mask = return_test_occ_mask

    @property
    def with_bbox(self):
        """bool: whether the RoI head contains a `bbox_head`"""
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_mask(self):
        """bool: whether the RoI head contains a `mask_head`"""
        return hasattr(self, 'mask_head') and self.mask_head is not None

    @property
    def with_front_mask(self):
        """bool: whether the RoI head contains a `mask_head`"""
        return hasattr(self, 'front_mask_head') and self.front_mask_head is not None

    @property
    def with_back_mask(self):
        """bool: whether the RoI head contains a `mask_head`"""
        return hasattr(self, 'back_mask_head') and self.back_mask_head is not None

    @property
    def with_shared_head(self):
        """bool: whether the RoI head contains a `shared_head`"""
        return hasattr(self, 'shared_head') and self.shared_head is not None

    @abstractmethod
    def init_weights(self, pretrained):
        """Initialize the weights in head.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        pass

    @abstractmethod
    def init_bbox_head(self):
        """Initialize ``bbox_head``"""
        pass

    @abstractmethod
    def init_mask_head(self):
        """Initialize ``mask_head``"""
        pass

    @abstractmethod
    def init_front_mask_head(self):
        """Initialize ``mask_head``"""
        pass

    @abstractmethod
    def init_back_mask_head(self):
        """Initialize ``mask_head``"""
        pass

    @abstractmethod
    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        pass

    @abstractmethod
    def forward_train(self,
                      x,
                      img_meta,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      **kwargs):
        """Forward function during training."""

    async def async_simple_test(self, x, img_meta, **kwargs):
        """Asynchronized test function."""
        raise NotImplementedError

    def simple_test(self,
                    x,
                    proposal_list,
                    img_meta,
                    proposals=None,
                    rescale=False,
                    **kwargs):
        """Test without augmentation."""

    def aug_test(self, x, proposal_list, img_metas, rescale=False, **kwargs):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
