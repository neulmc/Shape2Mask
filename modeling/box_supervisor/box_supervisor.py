from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..treat_shape2points import structured_loss, xielv_loss
from detectron2.layers import ShapeSpec, cat, ROIAlign
from detectron2.config import configurable
from detectron2.structures import Instances
from detectron2.utils.events import get_event_storage
from .target import create_box_targets, create_box_targets_p3, create_box_targets_crop, create_mask_targets_crop
from .utils import get_images_color_similarity, unfold_wo_center
from .ciou import ciou_loss
import numpy as np

class BoxSupervisor():
    def __init__(self, cfg):
        # box sup config
        if "Polygon" in cfg.MODEL.ROI_MASK_HEAD.NAME or "Shape" in cfg.MODEL.ROI_MASK_HEAD.NAME:
            mask_head_weights = cfg.MODEL.POLYGON_HEAD.POLY_LOSS.WS[0]
            is_logits = False
        else:
            mask_head_weights = cfg.MODEL.ROI_MASK_HEAD.MASK_HEAD_WEIGHTS 
            is_logits = True
        # box sup loss name
        boxsnake_loss = []
        if cfg.MODEL.BOX_SUP.LOSS_POINTS_SHAPE:
            boxsnake_loss.append('points_shape')
        if cfg.MODEL.BOX_SUP.LOSS_POINTS_RELA:
            boxsnake_loss.append('points_rela')
        if cfg.MODEL.BOX_SUP.LOSS_POINTS_PROJ:
            boxsnake_loss.append('points_projection')
        if cfg.MODEL.BOX_SUP.LOSS_LOCAL_PAIRWISE:
            boxsnake_loss.append("local_pairwise")
        if cfg.MODEL.BOX_SUP.LOSS_GLOBAL_PAIRWISE:
            boxsnake_loss.append("global_pairwise")
        self.seg_weights = cfg.MODEL.BOX_SUP.SEG_WEIGHTS
        print(f"box_sup_loss: {boxsnake_loss}")

        # box sup loss weights
        boxsnake_loss_weights = {
            "loss_points_shape": cfg.MODEL.BOX_SUP.LOSS_POINTS_SHAPE_WEIGHT,
            "loss_points_rela": cfg.MODEL.BOX_SUP.LOSS_POINTS_RELA_WEIGHT,
            "loss_points_proj": cfg.MODEL.BOX_SUP.LOSS_POINTS_PROJ_WEIGHT,
            "loss_local_pairwise": cfg.MODEL.BOX_SUP.LOSS_LOCAL_PAIRWISE_WEIGHT,
            "loss_global_pairwise": cfg.MODEL.BOX_SUP.LOSS_GLOBAL_PAIRWISE_WEIGHT,
            }
        boxsnake_loss_weights = {k: v * mask_head_weights for k, v in boxsnake_loss_weights.items()}
        print(f"box_sup_loss_weights: {boxsnake_loss_weights}")

        # local pairwise loss param
        self.enable_local_pairwise_loss = cfg.MODEL.BOX_SUP.LOSS_LOCAL_PAIRWISE
        self.pairwise_warmup_iters = cfg.MODEL.BOX_SUP.LOSS_PAIRWISE_WARMUP_ITER
        self.pairwise_cold_iters = cfg.MODEL.BOX_SUP.LOSS_PAIRWISE_COLD_ITER
        self.local_pairwise_kernel_size = cfg.MODEL.BOX_SUP.LOCAL_PAIRWISE_KERNEL_SIZE
        self.local_pairwise_dilation = cfg.MODEL.BOX_SUP.LOCAL_PAIRWISE_DILATION
        self.local_pairwise_color_threshold = cfg.MODEL.BOX_SUP.LOCAL_PAIRWISE_THR # for boxinst format pairwise
        self.local_pairwise_sigma = cfg.MODEL.BOX_SUP.LOCAL_PAIRWISE_SIGMA
        
        self.crop_predicts = cfg.MODEL.BOX_SUP.CROP_PREDICTS
        self.crop_size = cfg.MODEL.BOX_SUP.CROP_SIZE
        self.mask_padding_size = cfg.MODEL.BOX_SUP.MASK_PADDING_SIZE

        self.box_sup_loss = BoxSupLoss(
            boxsnake_loss, boxsnake_loss_weights, self.local_pairwise_kernel_size, 
            self.local_pairwise_dilation,
            is_logits=is_logits,
            loss_projection_type=cfg.MODEL.BOX_SUP.LOSS_PROJ_TYPE,
            local_pairwise_color_threshold=self.local_pairwise_color_threshold,
            loss_local_pairwise_type=cfg.MODEL.BOX_SUP.LOSS_LOCAL_PAIRWISE_TYPE,
            crop_predicts=self.crop_predicts,
            seg_weights = self.seg_weights,
            )
        
        self.mask_stride = cfg.MODEL.POLYGON_HEAD.MASK_STRIDE
        self.predict_in_box_space = cfg.MODEL.POLYGON_HEAD.PRED_WITHIN_BOX


    def __call__(self, images, preds, instances, pred_level):
        """
        images: dict()
        preds: dict()
        instances: list
        Return: loss
        """
        assert 'mask' in preds, "There should be mask in preds"
        pred_masks = preds["mask"]
        if not isinstance(pred_masks, list):
            pred_masks = [pred_masks]
        pred_polygons = preds.get("polygon", [torch.empty_like(pred_masks[0]) for i in range(len(pred_masks))])
        pred_shape = preds.get("shape", [torch.empty_like(pred_masks[0]) for i in range(len(pred_masks))])
        inst_classes = torch.cat([inst.gt_classes for inst in instances])
        roi_pred_scores = torch.cat([inst.roi_pred_scores for inst in instances])  #box1

        total_num_masks = pred_masks[0].size(0)
        # create targets
        if self.predict_in_box_space: # boundaryformer method, this method does not exploit the background label
            mask_side_len = pred_masks[0].size(2)
            clip_boxes = torch.cat([inst.proposal_boxes.tensor for inst in instances])
            tgt_masks, tgt_imgs_sim, tgt_imgs, gt_boxes = create_box_targets(
                images["images"], images["images_norm"], instances, clip_boxes=clip_boxes,
                mask_size=mask_side_len, kernel_size=self.local_pairwise_kernel_size,
                dilation=self.local_pairwise_dilation,
                sigma=self.local_pairwise_sigma,)
            tgt_masks = tgt_masks.unsqueeze(1).float()
        elif self.crop_predicts: # boxsnake take this method
            tgt_masks, tgt_imgs_sim, tgt_imgs, gt_boxes = create_box_targets_crop(
                images["images"], images["images_norm"], instances, self.mask_stride,
                kernel_size=self.local_pairwise_kernel_size,
                dilation=self.local_pairwise_dilation,
                crop_size=self.crop_size, # add to config
                mask_padding_size=self.mask_padding_size,
                sigma=self.local_pairwise_sigma,
            )
            tgt_masks = tgt_masks.unsqueeze(1).float()
        else:  # rasterizerize the polygon to p3/p2 
            tgt_masks, tgt_imgs_sim, tgt_imgs, gt_boxes = create_box_targets_p3(
                images["images"], images["images_norm"], instances, self.mask_stride,
                kernel_size=self.local_pairwise_kernel_size,
                dilation=self.local_pairwise_dilation,
                sigma=self.local_pairwise_sigma,
            )
            tgt_masks = tgt_masks.unsqueeze(1).float()
        
        del images
        
        targets = {"mask": tgt_masks, "imgs_sim": tgt_imgs_sim, "imgs": tgt_imgs, "gt_boxes": gt_boxes, "inst_classes":inst_classes}
        losses_list = []

        for i in range(len(pred_masks) - len(pred_shape)):
            pred_shape.append(None)
        for lid, (pred_masks_per_dec, pred_polys_per_dec, pred_shape_per_dec) in enumerate(zip(pred_masks, pred_polygons, pred_shape)):
            #if it is not the cls agnostic mask
            if pred_masks_per_dec.size(1) != 1:
                indices = torch.arange(total_num_masks)
                gt_classes = cat([inst.gt_classes.to(dtype=torch.int64) for inst in instances], dim=0)
                # shape=(N, num_class, mask_size, mask_size) -> (N, 1, mask_size, mask_size)
                pred_masks_per_dec = pred_masks_per_dec[indices, gt_classes]
                pred_masks_per_dec = pred_masks_per_dec.unsqueeze(1)
            # cal loss
            l_dict = self.box_sup_loss(pred_masks_per_dec, pred_polys_per_dec, pred_shape_per_dec, roi_pred_scores, targets, pred_level[lid])
            losses_list.append(l_dict)
        
        losses = {} # to average the same loss from different decoder
        for k in list(losses_list[0].keys()):
            losses[k] = torch.stack([d[k] for d in losses_list]).mean()

        return losses


class BoxSupLoss():
    def __init__(self, losses, loss_weights, local_pairwise_kernel_size=3, local_pairwise_dilation=1, 
                local_pairwise_color_threshold=0.1, loss_local_pairwise_type="v1", is_logits=False, 
                loss_projection_type=["dice"], crop_predicts=False, seg_weights = [1.0,1.0,1.0]):
        super(BoxSupLoss, self).__init__()
        self.losses = losses
        self.loss_weights = loss_weights
        
        # projection
        self.loss_projection_type = loss_projection_type
        print(f"loss_projection_type: {loss_projection_type}")
        loss_map = {
            'points_projection': self.loss_points_proj, # this is the CIoU loss
            'points_shape': self.loss_points_shape, # 先验损失
            'points_rela': self.loss_points_rela,
            'global_pairwise': self.loss_global_pairwise, # 图像约束
            'local_pairwise': self.loss_local_pairwise,
        }

        self.local_pairwise_kernel_size = local_pairwise_kernel_size
        self.local_pairwise_dilation = local_pairwise_dilation
        self.local_pairwise_color_threshold = local_pairwise_color_threshold # only for boxinst format pairiwse loss

        self.loss_map = {k: loss_map[k] for k in self.losses}
        self.crop_predicts = crop_predicts
        self.is_logits = is_logits
        self.seg_weights = torch.tensor(seg_weights, dtype=torch.float32)

    # 有监督损失
    def loss_points_proj(self, pred_masks, targets, num_masks, **kwargs):
        pred_polys = kwargs['pred_polys']
        gt_boxes = targets['gt_boxes']
        inst_classes = targets['inst_classes']
        # points proj
        proj_boxes = torch.cat([pred_polys.min(dim=1)[0], pred_polys.max(dim=1)[0]], dim=-1) # shape=(N, 4), x1y1x2y2 format
        # ciou loss
        loss = ciou_loss(proj_boxes, gt_boxes) * self.seg_weights.to(inst_classes.device)[inst_classes]
        return {'loss_points_proj': loss}

    # 先验损失，这个要用shape结尾 只对类别0/2有约束
    def loss_points_shape(self, pred_masks, targets, num_masks, **kwargs):
        WH_r = 4 #形状比例
        pred_shape = kwargs['pred_shape']
        roi_pred_scores = kwargs['roi_pred_scores']
        inst_classes = targets['inst_classes']
        roi_pred_scores = roi_pred_scores[torch.arange(roi_pred_scores.shape[0]), inst_classes]
        unique_classes = [0, 2]
        loss_class_consistency = torch.tensor(0., device=pred_shape.device)
        # 功能5 +6： 类别的形状
        for cls in unique_classes:
            mask = torch.where((inst_classes == cls) & (roi_pred_scores > 0))[0]  # 返回的是 CUDA Tensor
            if mask.any():
                if cls == 0:
                    ratios1 = pred_shape[mask, 0] / pred_shape[mask, 1]  # 计算宽高比
                    loss_class_consistency += torch.mean(torch.nn.functional.relu(WH_r - ratios1)* torch.sigmoid(roi_pred_scores[mask]).detach())
                elif cls == 2:
                    ratios1 = pred_shape[mask, 1] / pred_shape[mask, 0]  # 计算高宽比
                    loss_class_consistency += torch.mean(torch.nn.functional.relu(WH_r - ratios1)* torch.sigmoid(roi_pred_scores[mask]).detach())
            else:
                loss_class_consistency += torch.tensor(0., device=pred_shape.device)
        loss = loss_class_consistency
        loss = loss.mean() * self.loss_weights.get('loss_points_shape', 0.)
        return {'loss_points_shape': loss}

    # 这个要用rela 加点预测时的形状输出，这个要用shape结尾 只对类别0/2有约束
    def loss_points_rela(self, pred_masks, targets, num_masks, **kwargs):
        A = 0.1 # 斜率容忍度
        #A = 0.2  # 斜率容忍度
        pred_polys = kwargs['pred_polys']
        inst_classes = targets['inst_classes']
        mask_cls0 = (inst_classes == 0)
        mask_cls2 = (inst_classes == 2)
        loss_str, loss_xie = torch.tensor(0., device=pred_masks.device), torch.tensor(0., device=pred_masks.device)
        if mask_cls0.any():
            loss_str += structured_loss(pred_polys[mask_cls0]) # 这个是约束点的预测顺序的，不能乱序
            loss_xie += xielv_loss(pred_polys[mask_cls0], A) # 这个是约束尽量为在线条上的
        if mask_cls2.any():
            loss_str += structured_loss(pred_polys[mask_cls2])
            loss_xie += xielv_loss(pred_polys[mask_cls2], A) # 但是有的位置，生成的点正好回卡在画布的边上。。
        #sss  =loss_str + loss_xie
        #sss1 = torch.tensor(0., device=pred_masks.device)
        return {'loss_points_rela': (loss_str + loss_xie) * self.loss_weights.get('loss_points_rela', 0.)}

    # 图像损失1
    def loss_local_pairwise(self, pred_masks, targets, num_masks, **kwargs):
        # 这个函数实际上只对边缘像素起作用，只有边界像素才会出现邻域和自身预测类别不一致的情况
        target_masks, imgs_sim = targets["mask"], targets["imgs_sim"]
        inst_classes = targets['inst_classes']
        # fg_prob = torch.sigmoid(pred_masks) if self.is_logits else pred_masks
        fg_prob = pred_masks
        fg_prob_unfold = unfold_wo_center(
            fg_prob, kernel_size=self.local_pairwise_kernel_size,
            dilation=self.local_pairwise_dilation)
        pairwise_term = torch.abs(fg_prob[:, :, None] - fg_prob_unfold)[:, 0]
        weights = imgs_sim * target_masks.float()

        #ss1 = target_masks.detach().cpu().numpy()[12,0,:,:]
        #ss3 = target_masks.detach().cpu().numpy()[52,0,:,:]
        # limit to the box
        inst_weights = self.seg_weights.to(inst_classes.device)[inst_classes].view(-1, 1, 1, 1)
        loss_local_pairwise = (weights * pairwise_term * inst_weights).sum() / weights.sum().clamp(min=1.0)
        # TODO: which one ?
        # loss_local_pairwise = (weights * pairwise_term).flatten(1).sum(-1) / weights.flatten(1).sum(-1).clamp(min=1.0)
        # loss_local_pairwise = loss_local_pairwise.sum() / num_masks

        loss = {"loss_local_pairwise": loss_local_pairwise * self.loss_weights.get("loss_local_pairwise", 0.)}
        # TODO: add different pairwise format
        del target_masks
        del imgs_sim
        return loss

    # 图像损失2
    def loss_global_pairwise(self, pred_masks, targets, num_masks, **kwargs):
        target_masks, imgs = targets["mask"], targets["imgs"] # shape=(N, 1, H, W), (N, 3, H, W)
        inst_classes = targets['inst_classes']
        # prepare pred_masks
        pred_masks_back = 1.0 - pred_masks
        C_, H_, W_ = imgs.shape[1:]
        # imgs_wbox = imgs * target_masks # TODO, dose this matter in the cropped way?  
        level_set_energy = get_region_level_energy(imgs, pred_masks, C_) + \
                           get_region_level_energy(imgs, pred_masks_back, C_)

        pixel_num = float(H_ * W_)
        inst_weights = self.seg_weights.to(inst_classes.device)[inst_classes]
        level_set_losses = torch.mean((level_set_energy * inst_weights) / pixel_num) # HW weights
        losses = {"loss_global_pairwise": level_set_losses * self.loss_weights.get('loss_global_pairwise', 0.)} # instances weights

        del target_masks
        del imgs
        return losses

    def get_loss(self, loss, pred_masks, target_masks, num_masks, pred_level_now, **kwargs):
        assert loss in self.loss_map, f"do you really want to compute {loss} loss?"
        # 只对形状头函数进行约束，忽略其他
        if 'points_shape' in loss and pred_level_now != 'shape':
            return {'loss_' + loss: torch.tensor(0., device=pred_masks.device)}
        # 只对点或融合头函数进行约束，忽略shape头
        elif 'points_rela' in loss and pred_level_now == 'shape': #改lmc 之前最后一维是没有用到shape的，因为保存的是['shape']
            return {'loss_' + loss: torch.tensor(0., device=pred_masks.device)}
        else:
            return self.loss_map[loss](pred_masks, target_masks, num_masks, **kwargs)

    def __call__(self, pred_masks, pred_polys, pred_shape, roi_pred_scores, targets, pred_level_now):
        if self.is_logits:
            #pred_masks_ = pred_masks.detach().cpu().numpy()[0,0]
            pred_masks = pred_masks.sigmoid()
        num_masks = max(pred_masks.shape[0], 1.0)
        kwargs = {'pred_polys': pred_polys, 'pred_shape': pred_shape, 'roi_pred_scores': roi_pred_scores}
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, pred_masks, targets, num_masks, pred_level_now, **kwargs))
        return losses


class supBoxSupervisor():
    def __init__(self, cfg):
        # box sup config
        if "Polygon" in cfg.MODEL.ROI_MASK_HEAD.NAME or "Shape" in cfg.MODEL.ROI_MASK_HEAD.NAME:
            mask_head_weights = cfg.MODEL.POLYGON_HEAD.POLY_LOSS.WS[0]
            is_logits = False
        else:
            mask_head_weights = cfg.MODEL.ROI_MASK_HEAD.MASK_HEAD_WEIGHTS
            is_logits = True
        # box sup loss name
        boxsnake_loss = []
        if cfg.MODEL.BOX_SUP.LOSS_POINTS_SHAPE:
            boxsnake_loss.append('points_shape')
        if cfg.MODEL.BOX_SUP.LOSS_POINTS_RELA:
            boxsnake_loss.append('points_rela')
        if cfg.MODEL.BOX_SUP.LOSS_POINTS_PROJ:
            boxsnake_loss.append('points_projection')
        if cfg.MODEL.BOX_SUP.LOSS_LOCAL_PAIRWISE:
            boxsnake_loss.append("local_pairwise")
        if cfg.MODEL.BOX_SUP.LOSS_GLOBAL_PAIRWISE:
            boxsnake_loss.append("global_pairwise")
        self.seg_weights = cfg.MODEL.BOX_SUP.SEG_WEIGHTS
        print(f"box_sup_loss: {boxsnake_loss}")

        # box sup loss weights
        boxsnake_loss_weights = {
            "loss_points_shape": cfg.MODEL.BOX_SUP.LOSS_POINTS_SHAPE_WEIGHT,
            "loss_points_rela": cfg.MODEL.BOX_SUP.LOSS_POINTS_RELA_WEIGHT,
            "loss_points_proj": cfg.MODEL.BOX_SUP.LOSS_POINTS_PROJ_WEIGHT,
            "loss_local_pairwise": cfg.MODEL.BOX_SUP.LOSS_LOCAL_PAIRWISE_WEIGHT,
            "loss_global_pairwise": cfg.MODEL.BOX_SUP.LOSS_GLOBAL_PAIRWISE_WEIGHT,
        }
        boxsnake_loss_weights = {k: v * mask_head_weights for k, v in boxsnake_loss_weights.items()}
        print(f"box_sup_loss_weights: {boxsnake_loss_weights}")

        # local pairwise loss param
        self.enable_local_pairwise_loss = cfg.MODEL.BOX_SUP.LOSS_LOCAL_PAIRWISE
        self.pairwise_warmup_iters = cfg.MODEL.BOX_SUP.LOSS_PAIRWISE_WARMUP_ITER
        self.pairwise_cold_iters = cfg.MODEL.BOX_SUP.LOSS_PAIRWISE_COLD_ITER
        self.local_pairwise_kernel_size = cfg.MODEL.BOX_SUP.LOCAL_PAIRWISE_KERNEL_SIZE
        self.local_pairwise_dilation = cfg.MODEL.BOX_SUP.LOCAL_PAIRWISE_DILATION
        self.local_pairwise_color_threshold = cfg.MODEL.BOX_SUP.LOCAL_PAIRWISE_THR  # for boxinst format pairwise
        self.local_pairwise_sigma = cfg.MODEL.BOX_SUP.LOCAL_PAIRWISE_SIGMA

        self.crop_predicts = cfg.MODEL.BOX_SUP.CROP_PREDICTS
        self.crop_size = cfg.MODEL.BOX_SUP.CROP_SIZE
        self.mask_padding_size = cfg.MODEL.BOX_SUP.MASK_PADDING_SIZE

        self.box_sup_loss = supBoxSupLoss(
            boxsnake_loss, boxsnake_loss_weights, self.local_pairwise_kernel_size,
            self.local_pairwise_dilation,
            is_logits=is_logits,
            loss_projection_type=cfg.MODEL.BOX_SUP.LOSS_PROJ_TYPE,
            local_pairwise_color_threshold=self.local_pairwise_color_threshold,
            loss_local_pairwise_type=cfg.MODEL.BOX_SUP.LOSS_LOCAL_PAIRWISE_TYPE,
            crop_predicts=self.crop_predicts,
            seg_weights=self.seg_weights,
        )

        self.mask_stride = cfg.MODEL.POLYGON_HEAD.MASK_STRIDE
        self.predict_in_box_space = cfg.MODEL.POLYGON_HEAD.PRED_WITHIN_BOX

    def __call__(self, images, preds, instances, pred_level):
        """
        images: dict()
        preds: dict()
        instances: list
        Return: loss
        """
        assert 'mask' in preds, "There should be mask in preds"
        pred_masks = preds["mask"]
        if not isinstance(pred_masks, list):
            pred_masks = [pred_masks]
        pred_polygons = preds.get("polygon", [torch.empty_like(pred_masks[0]) for i in range(len(pred_masks))])
        pred_shape = preds.get("shape", [torch.empty_like(pred_masks[0]) for i in range(len(pred_masks))])
        inst_classes = torch.cat([inst.gt_classes for inst in instances])
        roi_pred_scores = torch.cat([inst.roi_pred_scores for inst in instances])  # box1

        total_num_masks = pred_masks[0].size(0)
        # create targets
        if self.predict_in_box_space:  # boundaryformer method, this method does not exploit the background label
            mask_side_len = pred_masks[0].size(2)
            clip_boxes = torch.cat([inst.proposal_boxes.tensor for inst in instances])
            tgt_masks, tgt_imgs_sim, tgt_imgs, gt_boxes = create_box_targets(
                images["images"], images["images_norm"], instances, clip_boxes=clip_boxes,
                mask_size=mask_side_len, kernel_size=self.local_pairwise_kernel_size,
                dilation=self.local_pairwise_dilation,
                sigma=self.local_pairwise_sigma, )
            tgt_masks = tgt_masks.unsqueeze(1).float()
        elif self.crop_predicts:  # boxsnake take this method
            tgt_masks, tgt_imgs_sim, tgt_imgs, gt_boxes, gt_realmasks = create_mask_targets_crop(
                images["images"], images["images_norm"], instances, self.mask_stride,
                kernel_size=self.local_pairwise_kernel_size,
                dilation=self.local_pairwise_dilation,
                crop_size=self.crop_size,  # add to config
                mask_padding_size=self.mask_padding_size,
                sigma=self.local_pairwise_sigma,
            )
            tgt_masks = tgt_masks.unsqueeze(1).float()
        else:  # rasterizerize the polygon to p3/p2
            tgt_masks, tgt_imgs_sim, tgt_imgs, gt_boxes = create_box_targets_p3(
                images["images"], images["images_norm"], instances, self.mask_stride,
                kernel_size=self.local_pairwise_kernel_size,
                dilation=self.local_pairwise_dilation,
                sigma=self.local_pairwise_sigma,
            )
            tgt_masks = tgt_masks.unsqueeze(1).float()

        del images

        targets = {"mask": tgt_masks, "imgs_sim": tgt_imgs_sim, "imgs": tgt_imgs, "gt_boxes": gt_boxes,
                   "inst_classes": inst_classes, "gt_realmasks": gt_realmasks}
        losses_list = []

        for i in range(len(pred_masks) - len(pred_shape)):
            pred_shape.append(None)
        for lid, (pred_masks_per_dec, pred_polys_per_dec, pred_shape_per_dec) in enumerate(
                zip(pred_masks, pred_polygons, pred_shape)):
            # if it is not the cls agnostic mask
            if pred_masks_per_dec.size(1) != 1:
                indices = torch.arange(total_num_masks)
                gt_classes = cat([inst.gt_classes.to(dtype=torch.int64) for inst in instances], dim=0)
                # shape=(N, num_class, mask_size, mask_size) -> (N, 1, mask_size, mask_size)
                pred_masks_per_dec = pred_masks_per_dec[indices, gt_classes]
                pred_masks_per_dec = pred_masks_per_dec.unsqueeze(1)
            # cal loss
            l_dict = self.box_sup_loss(pred_masks_per_dec, pred_polys_per_dec, pred_shape_per_dec, roi_pred_scores,
                                       targets, pred_level[lid])
            losses_list.append(l_dict)

        losses = {}  # to average the same loss from different decoder
        for k in list(losses_list[0].keys()):
            losses[k] = torch.stack([d[k] for d in losses_list]).mean()

        return losses


class supBoxSupLoss():
    def __init__(self, losses, loss_weights, local_pairwise_kernel_size=3, local_pairwise_dilation=1,
                 local_pairwise_color_threshold=0.1, loss_local_pairwise_type="v1", is_logits=False,
                 loss_projection_type=["dice"], crop_predicts=False, seg_weights=[1.0, 1.0, 1.0]):
        super(supBoxSupLoss, self).__init__()
        self.losses = losses
        self.loss_weights = loss_weights

        # projection
        self.loss_projection_type = loss_projection_type
        print(f"loss_projection_type: {loss_projection_type}")
        loss_map = {
            'points_projection': self.loss_points_proj,  # 新约束 supvised loss
            'points_shape': self.loss_points_shape,  # 先验损失
            'points_rela': self.loss_points_rela,
            'global_pairwise': self.loss_global_pairwise,  # 图像约束
            'local_pairwise': self.loss_local_pairwise,
        }

        self.local_pairwise_kernel_size = local_pairwise_kernel_size
        self.local_pairwise_dilation = local_pairwise_dilation
        self.local_pairwise_color_threshold = local_pairwise_color_threshold  # only for boxinst format pairiwse loss

        self.loss_map = {k: loss_map[k] for k in self.losses}
        self.crop_predicts = crop_predicts
        self.is_logits = is_logits
        self.seg_weights = torch.tensor(seg_weights, dtype=torch.float32)

    # 有监督损失
    def loss_points_proj(self, pred_masks, targets, num_masks, **kwargs):
        #pred_polys = kwargs['pred_polys']
        #gt_boxes = targets['gt_boxes']
        inst_classes = targets['inst_classes']
        gt_realmasks = targets['gt_realmasks']

        #gt_realmasks_ = gt_realmasks.detach().cpu().numpy()[4,0,:,:]
        #pred_masks_ = pred_masks.detach().cpu().numpy()[4,0,:,:]
        # points proj
        #proj_boxes = torch.cat([pred_polys.min(dim=1)[0], pred_polys.max(dim=1)[0]], dim=-1)  # shape=(N, 4), x1y1x2y2 format
        # ciou loss
        #loss = ciou_loss(proj_boxes, gt_boxes) * self.seg_weights.to(inst_classes.device)[inst_classes]

        gt_binary = (gt_realmasks > 0.01).float()
        inst_weights = self.seg_weights.to(inst_classes.device)[inst_classes]

        intersection = (pred_masks * gt_realmasks).sum(dim=(1, 2, 3))  # [B]
        union = pred_masks.sum(dim=(1, 2, 3)) + gt_realmasks.sum(dim=(1, 2, 3))  # [B]
        dice = (2 * intersection + 1e-6) / (union + 1e-6)  # [B]
        per_instance_loss = 1 - dice
        loss = (per_instance_loss * inst_weights).mean()* self.loss_weights.get('loss_points_proj', 0.)

        #per_pixel_loss = F.binary_cross_entropy(pred_masks, gt_binary, reduction='none')  # [B,1,H,W]
        #per_instance_loss = per_pixel_loss.mean(dim=(1, 2, 3))  # [B]
        #loss = (per_instance_loss * inst_weights).mean() * self.loss_weights.get('loss_points_proj', 0.)  # 加权平均

        return {'loss_points_proj': loss}

    # 先验损失，这个要用shape结尾 只对类别0/2有约束
    def loss_points_shape(self, pred_masks, targets, num_masks, **kwargs):
        WH_r = 4  # 形状比例
        pred_shape = kwargs['pred_shape']
        roi_pred_scores = kwargs['roi_pred_scores']
        inst_classes = targets['inst_classes']
        roi_pred_scores = roi_pred_scores[torch.arange(roi_pred_scores.shape[0]), inst_classes]
        unique_classes = [0, 2]
        loss_class_consistency = torch.tensor(0., device=pred_shape.device)
        # 功能5 +6： 类别的形状
        for cls in unique_classes:
            mask = torch.where((inst_classes == cls) & (roi_pred_scores > 0))[0]  # 返回的是 CUDA Tensor
            if mask.any():
                if cls == 0:
                    ratios1 = pred_shape[mask, 0] / pred_shape[mask, 1]  # 计算宽高比
                    loss_class_consistency += torch.mean(
                        torch.nn.functional.relu(WH_r - ratios1) * torch.sigmoid(roi_pred_scores[mask]).detach())
                elif cls == 2:
                    ratios1 = pred_shape[mask, 1] / pred_shape[mask, 0]  # 计算高宽比
                    loss_class_consistency += torch.mean(
                        torch.nn.functional.relu(WH_r - ratios1) * torch.sigmoid(roi_pred_scores[mask]).detach())
            else:
                loss_class_consistency += torch.tensor(0., device=pred_shape.device)
        loss = loss_class_consistency
        loss = loss.mean() * self.loss_weights.get('loss_points_shape', 0.)
        return {'loss_points_shape': loss}

    # 这个要用rela 加点预测时的形状输出，这个要用shape结尾 只对类别0/2有约束
    def loss_points_rela(self, pred_masks, targets, num_masks, **kwargs):
        A = 0.1  # 斜率容忍度
        # A = 0.2  # 斜率容忍度
        pred_polys = kwargs['pred_polys']
        inst_classes = targets['inst_classes']
        mask_cls0 = (inst_classes == 0)
        mask_cls2 = (inst_classes == 2)
        loss_str, loss_xie = torch.tensor(0., device=pred_masks.device), torch.tensor(0., device=pred_masks.device)
        if mask_cls0.any():
            loss_str += structured_loss(pred_polys[mask_cls0])  # 这个是约束点的预测顺序的，不能乱序
            loss_xie += xielv_loss(pred_polys[mask_cls0], A)  # 这个是约束尽量为在线条上的
        if mask_cls2.any():
            loss_str += structured_loss(pred_polys[mask_cls2])
            loss_xie += xielv_loss(pred_polys[mask_cls2], A)  # 但是有的位置，生成的点正好回卡在画布的边上。。
        # sss  =loss_str + loss_xie
        # sss1 = torch.tensor(0., device=pred_masks.device)
        return {'loss_points_rela': (loss_str + loss_xie) * self.loss_weights.get('loss_points_rela', 0.)}

    # 图像损失1
    def loss_local_pairwise(self, pred_masks, targets, num_masks, **kwargs):
        # 这个函数实际上只对边缘像素起作用，只有边界像素才会出现邻域和自身预测类别不一致的情况
        target_masks, imgs_sim = targets["mask"], targets["imgs_sim"]
        inst_classes = targets['inst_classes']
        # fg_prob = torch.sigmoid(pred_masks) if self.is_logits else pred_masks
        fg_prob = pred_masks
        fg_prob_unfold = unfold_wo_center(
            fg_prob, kernel_size=self.local_pairwise_kernel_size,
            dilation=self.local_pairwise_dilation)
        pairwise_term = torch.abs(fg_prob[:, :, None] - fg_prob_unfold)[:, 0]
        weights = imgs_sim * target_masks.float()

        # ss1 = target_masks.detach().cpu().numpy()[12,0,:,:]
        # ss3 = target_masks.detach().cpu().numpy()[52,0,:,:]
        # limit to the box
        inst_weights = self.seg_weights.to(inst_classes.device)[inst_classes].view(-1, 1, 1, 1)
        loss_local_pairwise = (weights * pairwise_term * inst_weights).sum() / weights.sum().clamp(min=1.0)
        # TODO: which one ?
        # loss_local_pairwise = (weights * pairwise_term).flatten(1).sum(-1) / weights.flatten(1).sum(-1).clamp(min=1.0)
        # loss_local_pairwise = loss_local_pairwise.sum() / num_masks

        loss = {"loss_local_pairwise": loss_local_pairwise * self.loss_weights.get("loss_local_pairwise", 0.)}
        # TODO: add different pairwise format
        del target_masks
        del imgs_sim
        return loss

    # 图像损失2
    def loss_global_pairwise(self, pred_masks, targets, num_masks, **kwargs):
        target_masks, imgs = targets["mask"], targets["imgs"]  # shape=(N, 1, H, W), (N, 3, H, W)
        inst_classes = targets['inst_classes']
        # prepare pred_masks
        pred_masks_back = 1.0 - pred_masks
        C_, H_, W_ = imgs.shape[1:]
        # imgs_wbox = imgs * target_masks # TODO, dose this matter in the cropped way?
        level_set_energy = get_region_level_energy(imgs, pred_masks, C_) + \
                           get_region_level_energy(imgs, pred_masks_back, C_)

        pixel_num = float(H_ * W_)
        inst_weights = self.seg_weights.to(inst_classes.device)[inst_classes]
        level_set_losses = torch.mean((level_set_energy * inst_weights) / pixel_num)  # HW weights
        losses = {"loss_global_pairwise": level_set_losses * self.loss_weights.get('loss_global_pairwise',
                                                                                   0.)}  # instances weights

        del target_masks
        del imgs
        return losses

    def get_loss(self, loss, pred_masks, target_masks, num_masks, pred_level_now, **kwargs):
        assert loss in self.loss_map, f"do you really want to compute {loss} loss?"
        # 只对形状头函数进行约束，忽略其他
        if 'points_shape' in loss and pred_level_now != 'shape':
            return {'loss_' + loss: torch.tensor(0., device=pred_masks.device)}
        # 只对点或融合头函数进行约束，忽略shape头
        elif 'points_rela' in loss and pred_level_now == 'shape':  # 改lmc 之前最后一维是没有用到shape的，因为保存的是['shape']
            return {'loss_' + loss: torch.tensor(0., device=pred_masks.device)}
        else:
            return self.loss_map[loss](pred_masks, target_masks, num_masks, **kwargs)

    def __call__(self, pred_masks, pred_polys, pred_shape, roi_pred_scores, targets, pred_level_now):
        if self.is_logits:
            # pred_masks_ = pred_masks.detach().cpu().numpy()[0,0]
            pred_masks = pred_masks.sigmoid()
        num_masks = max(pred_masks.shape[0], 1.0)
        kwargs = {'pred_polys': pred_polys, 'pred_shape': pred_shape, 'roi_pred_scores': roi_pred_scores}
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, pred_masks, targets, num_masks, pred_level_now, **kwargs))
        return losses

def get_region_level_energy(imgs, masks, num_channel):
    # masks = masks.expand(-1, num_channel, -1, -1)  # shape=(N, C, H, W) 不用 expand, 自己扩展就行
    avg_sim = torch.sum(imgs * masks, dim=(2, 3), keepdim=True) / torch.sum(masks, dim=(2, 3), keepdim=True).clamp(min=1e-5)
    # shape=(N, C, 1, 1)
    region_level = torch.pow(imgs - avg_sim, 2) * masks  # shape=(N, C, H, W), 沿着channel相加，沿着 HW 求和（积分）
    return torch.sum(region_level, dim=(1, 2, 3)) / num_channel
