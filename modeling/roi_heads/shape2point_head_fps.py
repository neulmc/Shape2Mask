import itertools
from typing import List
from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.modeling import ROI_MASK_HEAD_REGISTRY
from detectron2.structures import Boxes, BoxMode
from torch.nn.init import xavier_uniform_, constant_, normal_
from torch.nn.utils.rnn import pad_sequence
from modeling.layers.deform_attn.modules import MSDeformAttn
from modeling.poolers import MultiROIPooler
from modeling.position_encoding import build_position_encoding
from modeling.tensor import NestedTensor
from modeling.transformer import DeformableTransformerDecoder, DeformableTransformerControlLayer, MLP, \
    point_encoding, Shape_DeformableTransformerDecoder, shape_DeformableTransformerControlLayer
import numpy as np
from torch import nn
from torch.nn import functional as F
from detectron2.structures.instances import Instances
from modeling.layers.diff_ras.polygon import SoftPolygon
from modeling.box_supervisor import BoxSupervisor, supBoxSupervisor
from modeling.criterion import build_poly_losses
from modeling.criterion import MaskCriterion
import torch
import math
from modeling.treat_shape2points import batch_get_rotated_rect_vertices, pad_inst_classes, batch_get_rotated_rect_vertices_points
from modeling.treat_shape2points import modify_base_shape
from ..merge_shapa_point import TransformerFusion

@ROI_MASK_HEAD_REGISTRY.register()
class Shape2pointHead(nn.Module):
    """
    polygon head from BoundaryFormer
    """

    @configurable
    def __init__(self, input_shape: ShapeSpec, in_features, vertex_loss_fns, vertex_loss_ws, mask_criterion,
                 box_supervisor, ref_init="ellipse",
                 model_dim=256, base_number_control_points=8, number_control_points=64, vis_period=0,
                 is_upsampling=True, iterative_refinement=False, use_p2p_attn=True, num_classes=80,
                 cls_agnostic=False,
                 predict_in_box_space=False, prepool=True, dropout=0.0, deep_supervision=True,
                 inv_smoothness=0.1, resolution_list=[], enable_box_sup=False, box_feat_pooler=None,
                 box_feat_refiner=None, mask_stride=8,
                 crop_predicts=False, crop_size=64, mask_padding_size=4,
                 idx_output=None, shape_point_num= 8, shape_para = [1, 1, 0.5, 0.5, 0.5],
                 whratio = 0.2, KAN = True, MLP_KAN = True, use_cls_token=True,
                 shape_number_layers=2, point_number_layers=2, merge = False, merge_weight = 0.1,
                 decode_gat = False, fusion_gat = False, cross_atten_points = 4, device = None, Kan_loss=0, **kwargs):
        super().__init__()

        self.input_shape = input_shape
        self.in_features = in_features
        self.num_feature_levels = len(self.in_features)
        self.ref_init = ref_init
        self.shape_point_num = shape_point_num
        self.shape_para = shape_para # width, height, center_x, center_y, angle
        self.whratio = whratio
        self.MLP_KAN = MLP_KAN
        self.KAN = KAN
        self.batch_size_div = 16
        self.device = device
        self.Kan_loss = Kan_loss
        if Kan_loss == 0:
            self.Kan_loss_used = False
        else:
            self.Kan_loss_used = True

        if not ref_init in ["ellipse", "random", "convex", "square"]:
            raise ValueError("unknown ref_init {0}".format(ref_init))

        self.base_number_control_points = base_number_control_points
        self.number_control_points = number_control_points
        self.model_dimension = model_dim
        self.is_upsampling = is_upsampling
        self.iterative_refinement = iterative_refinement or self.is_upsampling
        self.use_cls_token = use_cls_token
        self.use_p2p_attn = use_p2p_attn
        self.num_classes = num_classes
        self.cls_agnostic = cls_agnostic
        self.vis_period = vis_period
        self.predict_in_box_space = predict_in_box_space
        self.crop_predicts = crop_predicts
        self.prepool = prepool
        self.dropout = dropout
        self.deep_supervision = deep_supervision
        if self.use_cls_token:
            self.shape_class_embedding_layer = nn.Embedding(num_classes, self.model_dimension)
            self.point_class_embedding_layer = nn.Embedding(num_classes, self.model_dimension)

        self.vertex_loss_fns = []
        for loss_fn in vertex_loss_fns:
            loss_fn_attr_name = "vertex_loss_fn_{0}".format(loss_fn.name)
            self.add_module(loss_fn_attr_name, loss_fn)

            self.vertex_loss_fns.append(getattr(self, loss_fn_attr_name))

        # add each as a module so it gets moved to the right device.
        self.vertex_loss_ws = vertex_loss_ws

        if len(self.vertex_loss_fns) != len(self.vertex_loss_ws):
            raise ValueError("vertex loss mismatch")

        self.position_embedding = build_position_encoding(self.model_dimension, kind="sine")
        self.level_embed = nn.Embedding(self.num_feature_levels, self.model_dimension)
        self.register_buffer("point_embedding", point_encoding(self.model_dimension * 2, max_len=self.number_control_points))
        self.register_buffer("shape_embedding", point_encoding(self.model_dimension * 2, max_len=self.shape_point_num))
        #self.point_embedding = point_encoding(self.model_dimension * 2, max_len=self.number_control_points).to(self.device)
        #self.shape_embedding = point_encoding(self.model_dimension * 2, max_len=self.shape_point_num).to(self.device)

        if self.ref_init == "random":
            self.reference_points = nn.Linear(self.model_dimension, 2)

        self.feature_proj = None
        self.merge = merge
        if self.merge:
            self.fusion = TransformerFusion(feature_dim = self.model_dimension, merge_weight = merge_weight, fusion_gat = fusion_gat)

        activation = "relu"
        nhead = 8

        self.feedforward_dimension = 1024
        decoder_layer = []
        shape_decoder_layer = []
        for _ in range(point_number_layers):
            decoder_layer.append(DeformableTransformerControlLayer(
                self.model_dimension, self.feedforward_dimension, self.dropout, activation, self.num_feature_levels,
                nhead, cross_atten_points, use_p2p_attn=self.use_p2p_attn, decode_gat=decode_gat, device=self.device))
        for  _ in range(shape_number_layers):
            shape_decoder_layer.append(shape_DeformableTransformerControlLayer(
                self.model_dimension, self.feedforward_dimension, self.dropout, activation, self.num_feature_levels,
                nhead, cross_atten_points, use_p2p_attn=self.use_p2p_attn, decode_gat=decode_gat, device=self.device))

        self.shape_number_layers = shape_number_layers
        self.point_number_layers = point_number_layers

        self.shape_decode = Shape_DeformableTransformerDecoder(
            nn.ModuleList(shape_decoder_layer), shape_number_layers, self.model_dimension, self.cls_agnostic, self.num_classes, self.MLP_KAN, return_intermediate=True, predict_in_box_space=self.predict_in_box_space)
        self.decoder = DeformableTransformerDecoder(
            nn.ModuleList(decoder_layer), point_number_layers, self.model_dimension, self.cls_agnostic, self.num_classes, self.MLP_KAN, return_intermediate=True, predict_in_box_space=self.predict_in_box_space)

        # rasterizer
        self.inv_smoothness = inv_smoothness
        self.offset = 0.5
        self.pred_rasterizer = SoftPolygon(inv_smoothness=self.inv_smoothness, mode="mask")
        # self.pred_rasterizer = SoftPolygonBatch(inv_smoothness=self.inv_smoothness)
        self.register_buffer("rasterize_at", torch.from_numpy(np.array(resolution_list).reshape(-1, 2))) # lmc
        #self.rasterize_at = torch.from_numpy(np.array(resolution_list).reshape(-1, 2)).to(self.device) # lmc

        mask_criterion.rasterize_at = self.rasterize_at
        self.mask_criterion = mask_criterion

        self.mask_stride = mask_stride
        self.mask_stride_lvl_name = f'p{str(int(math.log(mask_stride, 2)))}'
        assert self.mask_stride_lvl_name in self.in_features

        # box_sup
        self.enable_box_sup = enable_box_sup
        self.box_supervisor = box_supervisor
        self.box_feat_pooler = box_feat_pooler
        self.box_feat_refiner = box_feat_refiner
        self.mask_padding_size = mask_padding_size
        self.crop_size = crop_size

        # inference
        self.idx_output = -1
        #self.idx_output = 3
        self.debug = False

        self._reset_parameters()

    def _reset_parameters(self):
        for name, p in self.named_parameters():
            if ("fusion.weight_proj" in name):
                continue
            if ("shape_decode.shape_xy_embed" in name):
                continue
            if ("decoder.xy_embed" in name):
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

        if self.ref_init == "random":
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)

        normal_(self.level_embed.weight.data)

    @classmethod
    def from_config(cls, cfg, input_shape):
        in_features = cfg.MODEL.POLYGON_HEAD.IN_FEATURES
        enable_box_sup = cfg.MODEL.BOX_SUP.ENABLE

        ret = {
            "in_features": in_features,
            "ref_init": cfg.MODEL.POLYGON_HEAD.POLY_INIT,
            "model_dim": cfg.MODEL.POLYGON_HEAD.MODEL_DIM,
            "point_number_layers": cfg.MODEL.POLYGON_HEAD.POINT_NUM_DEC_LAYERS,
            "shape_number_layers": cfg.MODEL.POLYGON_HEAD.SHAPE_NUM_DEC_LAYERS,
            "base_number_control_points": cfg.MODEL.POLYGON_HEAD.UPSAMPLING_BASE_NUM_PTS,
            "number_control_points": cfg.MODEL.POLYGON_HEAD.POLY_NUM_PTS,
            "vis_period": cfg.VIS_PERIOD,
            "vertex_loss_fns": build_poly_losses(cfg, input_shape),
            "vertex_loss_ws": cfg.MODEL.POLYGON_HEAD.POLY_LOSS.WS,
            "mask_criterion": MaskCriterion(cfg),
            "box_supervisor": BoxSupervisor(cfg), # 第一次调用， 第二次调用
            "is_upsampling": cfg.MODEL.POLYGON_HEAD.UPSAMPLING,
            "iterative_refinement": cfg.MODEL.POLYGON_HEAD.ITER_REFINE,
            "use_cls_token": cfg.MODEL.POLYGON_HEAD.USE_CLS_TOKEN,
            "use_p2p_attn": cfg.MODEL.POLYGON_HEAD.USE_P2P_ATTN,
            "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "cls_agnostic": cfg.MODEL.POLYGON_HEAD.CLS_AGNOSTIC_MASK,
            "predict_in_box_space": cfg.MODEL.POLYGON_HEAD.PRED_WITHIN_BOX,
            "prepool": cfg.MODEL.POLYGON_HEAD.PREPOOL,
            "dropout": cfg.MODEL.POLYGON_HEAD.DROPOUT,
            "deep_supervision": cfg.MODEL.POLYGON_HEAD.DEEP_SUPERVISION,
            "inv_smoothness": cfg.MODEL.DIFFRAS.INV_SMOOTHNESS_SCHED[0],
            "resolution_list": cfg.MODEL.DIFFRAS.RESOLUTIONS,
            "enable_box_sup": enable_box_sup,
            "mask_stride": cfg.MODEL.POLYGON_HEAD.MASK_STRIDE,
            "crop_predicts": cfg.MODEL.BOX_SUP.CROP_PREDICTS,
            "crop_size": cfg.MODEL.BOX_SUP.CROP_SIZE,
            "mask_padding_size": cfg.MODEL.BOX_SUP.MASK_PADDING_SIZE,
            # lmc
            "shape_point_num": cfg.MODEL.POLYGON_HEAD.shape_point_num,  # lmc
            "shape_para": cfg.MODEL.POLYGON_HEAD.shape_para,  # lmc
            "whratio": cfg.MODEL.POLYGON_HEAD.whratio,  # lmc
            "KAN": cfg.MODEL.POLYGON_HEAD.KAN,  # lmc
            "MLP_KAN": cfg.MODEL.POLYGON_HEAD.MLP_KAN,  # lmc
            "merge": cfg.MODEL.POLYGON_HEAD.MERGE,
            "merge_weight": cfg.MODEL.POLYGON_HEAD.MERGE_WEIGHT,
            "fusion_gat": cfg.MODEL.POLYGON_HEAD.FUSION_GAT,
            "decode_gat": cfg.MODEL.POLYGON_HEAD.DECODE_GAT,
            "cross_atten_points": cfg.MODEL.POLYGON_HEAD.CROSS_POINTS,
            "device": cfg.MODEL.DEVICE,
            "Kan_loss": cfg.MODEL.LOSS_KAN_REG,
        }

        ret.update(
            input_shape=input_shape,
        )
        return ret

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def shape_get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_w, valid_ratio_w, valid_ratio_w, valid_ratio_w], -1)
        return valid_ratio

    def build_base_shape(self, batch_size, max_instances, instances, device):
        base_shape = torch.tensor(self.shape_para, dtype=torch.float32, device=device)  # shape: [5]
        base_shape = base_shape[None, None, :].expand(batch_size, max_instances, -1).unsqueeze(2)
        base_shape = base_shape.detach().clone().requires_grad_(base_shape.requires_grad)
        inst_classes_shape = pad_inst_classes(instances, max_instances, self.training, device)
        inst_classes = torch.cat([inst.gt_classes if self.training else inst.pred_classes for inst in instances])
        base_shape = modify_base_shape(base_shape, inst_classes_shape, ratio = self.whratio)
        return base_shape, inst_classes, inst_classes_shape

    def catch_embenddins(self, x, number_levels, batch_size, instances, device):
        masks = []
        pos_embeds = []
        srcs = []

        for l in range(number_levels):
            srcs.append(x[l])
            mask = torch.zeros((batch_size, x[l].shape[-2], x[l].shape[-1]), dtype=torch.bool, device=device)
            masks.append(mask)
            # todo, for non-pooled situation.. actually get the mask.
            f = NestedTensor(x[l], mask)
            pos_embeds.append(self.position_embedding(f))

        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed.weight[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)

        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        number_instances = [len(inst) for inst in instances]
        max_instances = max(number_instances)

        box_preds_xyxy = pad_sequence([
            (inst.proposal_boxes.tensor if self.training else inst.pred_boxes.tensor) / torch.Tensor(2 * inst.image_size[::-1]).to(device)
            for inst in instances], batch_first=True)

        return src_flatten, mask_flatten, lvl_pos_embed_flatten, level_start_index, valid_ratios, spatial_shapes, \
            number_instances, max_instances, box_preds_xyxy

    def forward(self, images, x, instances: List[Instances]):
        # 推理时跳过不必要的计算
        if not self.training:
            return self.fast_inference(images, x, instances)

        # 训练路径保持不变
        # ... 原有训练代码 ...

    @torch.no_grad()
    def fast_inference(self, images, x, instances: List[Instances]):
        """优化的推理路径"""
        device = x['p2'].device
        mask_wh = torch.as_tensor(x[self.mask_stride_lvl_name].shape[-2:][::-1], device=device)
        x = [x[f] for f in self.in_features]

        # 简化prepool（如果必须使用）
        if self.prepool:
            aligner = MultiROIPooler(
                [tuple(x_.shape[-2:]) for x_ in x],
                scales=(0.25, 0.125, 0.0625, 0.03125),
                sampling_ratio=0,
                pooler_type="ROIAlignV2",
                assign_to_single_level=False)

            roi_boxes = [Boxes(torch.tensor([[0, 0, inst.image_size[1], inst.image_size[0]]], device=device))
                         for inst in instances]
            x = aligner(x, roi_boxes)

        # 快速的特征准备
        batch_size = x[0].shape[0]
        src_flatten, mask_flatten, lvl_pos_embed_flatten, level_start_index, valid_ratios, spatial_shapes, \
            number_instances, max_instances, box_preds_xyxy = self.catch_embenddins(
            x, len(x), batch_size, instances, device)

        # 只计算必要的解码层输出
        memory = src_flatten + lvl_pos_embed_flatten

        # 快速shape解码（只取最终输出）
        if self.use_cls_token:
            inst_classes_shape = pad_inst_classes(instances, max_instances, self.training, device)
            shape_cls_embed = self.shape_class_embedding_layer(inst_classes_shape).unsqueeze(2).expand(-1, -1,
                                                                                                       self.shape_point_num,
                                                                                                       -1)
            point_cls_embed = self.point_class_embedding_layer(inst_classes_shape).unsqueeze(2).expand(-1, -1,
                                                                                                       self.shape_point_num,
                                                                                                       -1)

        shape_query_embed, shape_tgt = torch.split(self.shape_embedding, self.model_dimension, dim=1)
        shape_query_embed = shape_query_embed.unsqueeze(0).expand(batch_size, max_instances, -1, -1)
        shape_tgt = shape_tgt.unsqueeze(0).expand(batch_size, max_instances, -1, -1)

        base_shape, inst_classes, inst_classes_shape = self.build_base_shape(batch_size, max_instances, instances,
                                                                             device)
        if not self.predict_in_box_space:
            box_preds_xywh = [BoxMode.convert((inst.proposal_boxes.tensor if self.training else inst.pred_boxes.tensor),
                                              BoxMode.XYXY_ABS, BoxMode.XYWH_ABS) for inst in instances] # shape=(B, N, 4)
            padded_box_preds_xywh = pad_sequence(box_preds_xywh, batch_first=True)
            pad_img_wh = mask_wh * self.mask_stride  # devide
            # 这一步是将框选内固定的位置，转换成图像中的比例
            padded_box_x, padded_box_y, padded_box_w, padded_box_h = torch.split(padded_box_preds_xywh, 1, dim=-1)
            base_shape[:, :, :, 0] = base_shape[:,:,:,0] * padded_box_w / pad_img_wh[0]
            base_shape[:, :, :, 1] = base_shape[:,:,:,1] * padded_box_h / pad_img_wh[1]
            base_shape[:, :, :, 2] = (base_shape[:,:,:,2] * padded_box_w + padded_box_x) / pad_img_wh[0]
            base_shape[:, :, :, 3] = (base_shape[:,:,:,3] * padded_box_h + padded_box_y) / pad_img_wh[1]
        # 只获取最终输出，跳过中间层
        shape_intermediate, shape_outputs_coords, _ = self.shape_decode(
            (shape_tgt + shape_cls_embed), base_shape, memory, spatial_shapes, level_start_index,
            valid_ratios, shape_query_embed, mask_flatten, cls_token=None,
            reference_boxes=box_preds_xyxy, inst_classes_shape=inst_classes_shape)
        shape_outputs_coords_final = shape_outputs_coords[-1]
        for i in range(len(shape_outputs_coords)):
            output_coords = shape_outputs_coords[i]
            shape_outputs_coords[i] = torch.cat([output_coords[j, :number_instances[j]] for j in range(batch_size)])
            output_features = shape_intermediate[i]
            shape_intermediate[i] = torch.cat([output_features[j, :number_instances[j]] for j in range(batch_size)])
        shape_outputs_points = []
        for i in range(len(shape_outputs_coords)):#ss = instances[0].pred_classes
            shape_outputs_points.append(batch_get_rotated_rect_vertices(shape_outputs_coords[i], inst_classes))

        # 快速point解码
        query_embed, tgt = torch.split(self.point_embedding, self.model_dimension, dim=1)
        query_embed = query_embed.unsqueeze(0).expand(batch_size, max_instances, -1, -1)
        tgt = tgt.unsqueeze(0).expand(batch_size, max_instances, -1, -1)

        reference_points_real = batch_get_rotated_rect_vertices_points(shape_outputs_coords_final, inst_classes_shape)
        point_intermediate, outputs_coords, _ = self.decoder(
            (tgt + point_cls_embed), reference_points_real, memory, spatial_shapes, level_start_index,
            valid_ratios, query_embed, mask_flatten, cls_token=None, reference_boxes=box_preds_xyxy)
        for i in range(len(outputs_coords)):
            output_coords = outputs_coords[i]
            outputs_coords[i] = torch.cat([output_coords[j, :number_instances[j]] for j in range(batch_size)])
            output_features = point_intermediate[i]
            point_intermediate[i] = torch.cat([output_features[j, :number_instances[j]] for j in range(batch_size)])

        outputs_coords = shape_outputs_points + outputs_coords
        point_intermediate = shape_intermediate + point_intermediate
        if self.merge:
            outputs_coord_f, shape_point_weights = self.fusion(outputs_coords, point_intermediate)
            outputs_coords.append(outputs_coord_f)

        # 只取最终输出
        final_output = outputs_coords[-1]  # 形状应该是 [batch_size, max_instances, num_points, 2]

        # 修复split操作 - 根据实际形状调整
        # final_output 形状: [batch_size, max_instances, num_points, 2]
        # 我们需要按实际实例数量分割

        # 方法1: 直接处理每个batch
        pad_img_wh = mask_wh * self.mask_stride
        inst_now = 0

        for j in range(batch_size):
            num_real_instances = number_instances[j]
            if num_real_instances == 0:
                continue

            # 获取该batch的实际实例预测
            #batch_pred_polys = final_output[j, :num_real_instances]  # [num_real_instances, num_points, 2]
            batch_pred_polys = final_output[inst_now:inst_now+num_real_instances]
            inst_now = inst_now+num_real_instances

            # 根据配置调整坐标
            if not self.predict_in_box_space:
                batch_pred_polys = batch_pred_polys * pad_img_wh

            # 分配给对应的instance
            instances[j].pred_polys = batch_pred_polys

        return instances, None

@ROI_MASK_HEAD_REGISTRY.register()
class supShape2pointHead(nn.Module):
    """
    polygon head from BoundaryFormer
    """

    @configurable
    def __init__(self, input_shape: ShapeSpec, in_features, vertex_loss_fns, vertex_loss_ws, mask_criterion,
                 box_supervisor, ref_init="ellipse",
                 model_dim=256, base_number_control_points=8, number_control_points=64, vis_period=0,
                 is_upsampling=True, iterative_refinement=False, use_p2p_attn=True, num_classes=80,
                 cls_agnostic=False,
                 predict_in_box_space=False, prepool=True, dropout=0.0, deep_supervision=True,
                 inv_smoothness=0.1, resolution_list=[], enable_box_sup=False, box_feat_pooler=None,
                 box_feat_refiner=None, mask_stride=8,
                 crop_predicts=False, crop_size=64, mask_padding_size=4,
                 idx_output=None, shape_point_num= 8, shape_para = [1, 1, 0.5, 0.5, 0.5],
                 whratio = 0.2, KAN = True, MLP_KAN = True, use_cls_token=True,
                 shape_number_layers=2, point_number_layers=2, merge = False, merge_weight = 0.1,
                 decode_gat = False, fusion_gat = False, cross_atten_points = 4, device = None, Kan_loss=0, **kwargs):
        super().__init__()

        self.input_shape = input_shape
        self.in_features = in_features
        self.num_feature_levels = len(self.in_features)
        self.ref_init = ref_init
        self.shape_point_num = shape_point_num
        self.shape_para = shape_para # width, height, center_x, center_y, angle
        self.whratio = whratio
        self.MLP_KAN = MLP_KAN
        self.KAN = KAN
        self.batch_size_div = 16
        self.device = device
        self.Kan_loss = Kan_loss
        if Kan_loss == 0:
            self.Kan_loss_used = False
        else:
            self.Kan_loss_used = True

        if not ref_init in ["ellipse", "random", "convex", "square"]:
            raise ValueError("unknown ref_init {0}".format(ref_init))

        self.base_number_control_points = base_number_control_points
        self.number_control_points = number_control_points
        self.model_dimension = model_dim
        self.is_upsampling = is_upsampling
        self.iterative_refinement = iterative_refinement or self.is_upsampling
        self.use_cls_token = use_cls_token
        self.use_p2p_attn = use_p2p_attn
        self.num_classes = num_classes
        self.cls_agnostic = cls_agnostic
        self.vis_period = vis_period
        self.predict_in_box_space = predict_in_box_space
        self.crop_predicts = crop_predicts
        self.prepool = prepool
        self.dropout = dropout
        self.deep_supervision = deep_supervision
        if self.use_cls_token:
            self.shape_class_embedding_layer = nn.Embedding(num_classes, self.model_dimension)
            self.point_class_embedding_layer = nn.Embedding(num_classes, self.model_dimension)

        self.vertex_loss_fns = []
        for loss_fn in vertex_loss_fns:
            loss_fn_attr_name = "vertex_loss_fn_{0}".format(loss_fn.name)
            self.add_module(loss_fn_attr_name, loss_fn)

            self.vertex_loss_fns.append(getattr(self, loss_fn_attr_name))

        # add each as a module so it gets moved to the right device.
        self.vertex_loss_ws = vertex_loss_ws

        if len(self.vertex_loss_fns) != len(self.vertex_loss_ws):
            raise ValueError("vertex loss mismatch")

        self.position_embedding = build_position_encoding(self.model_dimension, kind="sine")
        self.level_embed = nn.Embedding(self.num_feature_levels, self.model_dimension)
        self.register_buffer("point_embedding", point_encoding(self.model_dimension * 2, max_len=self.number_control_points))
        self.register_buffer("shape_embedding", point_encoding(self.model_dimension * 2, max_len=self.shape_point_num))
        #self.point_embedding = point_encoding(self.model_dimension * 2, max_len=self.number_control_points).to(self.device)
        #self.shape_embedding = point_encoding(self.model_dimension * 2, max_len=self.shape_point_num).to(self.device)

        if self.ref_init == "random":
            self.reference_points = nn.Linear(self.model_dimension, 2)

        self.feature_proj = None
        self.merge = merge
        if self.merge:
            self.fusion = TransformerFusion(feature_dim = self.model_dimension, merge_weight = merge_weight, fusion_gat = fusion_gat)

        activation = "relu"
        nhead = 8

        self.feedforward_dimension = 1024
        decoder_layer = []
        shape_decoder_layer = []
        for _ in range(point_number_layers):
            decoder_layer.append(DeformableTransformerControlLayer(
                self.model_dimension, self.feedforward_dimension, self.dropout, activation, self.num_feature_levels,
                nhead, cross_atten_points, use_p2p_attn=self.use_p2p_attn, decode_gat=decode_gat, device=self.device))
        for  _ in range(shape_number_layers):
            shape_decoder_layer.append(shape_DeformableTransformerControlLayer(
                self.model_dimension, self.feedforward_dimension, self.dropout, activation, self.num_feature_levels,
                nhead, cross_atten_points, use_p2p_attn=self.use_p2p_attn, decode_gat=decode_gat, device=self.device))

        self.shape_number_layers = shape_number_layers
        self.point_number_layers = point_number_layers

        self.shape_decode = Shape_DeformableTransformerDecoder(
            nn.ModuleList(shape_decoder_layer), shape_number_layers, self.model_dimension, self.cls_agnostic, self.num_classes, self.MLP_KAN, return_intermediate=True, predict_in_box_space=self.predict_in_box_space)
        self.decoder = DeformableTransformerDecoder(
            nn.ModuleList(decoder_layer), point_number_layers, self.model_dimension, self.cls_agnostic, self.num_classes, self.MLP_KAN, return_intermediate=True, predict_in_box_space=self.predict_in_box_space)

        # rasterizer
        self.inv_smoothness = inv_smoothness
        self.offset = 0.5
        self.pred_rasterizer = SoftPolygon(inv_smoothness=self.inv_smoothness, mode="mask")
        # self.pred_rasterizer = SoftPolygonBatch(inv_smoothness=self.inv_smoothness)
        self.register_buffer("rasterize_at", torch.from_numpy(np.array(resolution_list).reshape(-1, 2))) # lmc
        #self.rasterize_at = torch.from_numpy(np.array(resolution_list).reshape(-1, 2)).to(self.device) # lmc

        mask_criterion.rasterize_at = self.rasterize_at
        self.mask_criterion = mask_criterion

        self.mask_stride = mask_stride
        self.mask_stride_lvl_name = f'p{str(int(math.log(mask_stride, 2)))}'
        assert self.mask_stride_lvl_name in self.in_features

        # box_sup
        self.enable_box_sup = enable_box_sup
        self.box_supervisor = box_supervisor
        self.box_feat_pooler = box_feat_pooler
        self.box_feat_refiner = box_feat_refiner
        self.mask_padding_size = mask_padding_size
        self.crop_size = crop_size

        # inference
        self.idx_output = -1
        #self.idx_output = 3
        self.debug = False

        self._reset_parameters()

    def _reset_parameters(self):
        for name, p in self.named_parameters():
            if ("fusion.weight_proj" in name):
                continue
            if ("shape_decode.shape_xy_embed" in name):
                continue
            if ("decoder.xy_embed" in name):
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

        if self.ref_init == "random":
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)

        normal_(self.level_embed.weight.data)

    @classmethod
    def from_config(cls, cfg, input_shape):
        in_features = cfg.MODEL.POLYGON_HEAD.IN_FEATURES
        enable_box_sup = cfg.MODEL.BOX_SUP.ENABLE

        ret = {
            "in_features": in_features,
            "ref_init": cfg.MODEL.POLYGON_HEAD.POLY_INIT,
            "model_dim": cfg.MODEL.POLYGON_HEAD.MODEL_DIM,
            "point_number_layers": cfg.MODEL.POLYGON_HEAD.POINT_NUM_DEC_LAYERS,
            "shape_number_layers": cfg.MODEL.POLYGON_HEAD.SHAPE_NUM_DEC_LAYERS,
            "base_number_control_points": cfg.MODEL.POLYGON_HEAD.UPSAMPLING_BASE_NUM_PTS,
            "number_control_points": cfg.MODEL.POLYGON_HEAD.POLY_NUM_PTS,
            "vis_period": cfg.VIS_PERIOD,
            "vertex_loss_fns": build_poly_losses(cfg, input_shape),
            "vertex_loss_ws": cfg.MODEL.POLYGON_HEAD.POLY_LOSS.WS,
            "mask_criterion": MaskCriterion(cfg),
            "box_supervisor": supBoxSupervisor(cfg), # 第一次调用， 第二次调用
            "is_upsampling": cfg.MODEL.POLYGON_HEAD.UPSAMPLING,
            "iterative_refinement": cfg.MODEL.POLYGON_HEAD.ITER_REFINE,
            "use_cls_token": cfg.MODEL.POLYGON_HEAD.USE_CLS_TOKEN,
            "use_p2p_attn": cfg.MODEL.POLYGON_HEAD.USE_P2P_ATTN,
            "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "cls_agnostic": cfg.MODEL.POLYGON_HEAD.CLS_AGNOSTIC_MASK,
            "predict_in_box_space": cfg.MODEL.POLYGON_HEAD.PRED_WITHIN_BOX,
            "prepool": cfg.MODEL.POLYGON_HEAD.PREPOOL,
            "dropout": cfg.MODEL.POLYGON_HEAD.DROPOUT,
            "deep_supervision": cfg.MODEL.POLYGON_HEAD.DEEP_SUPERVISION,
            "inv_smoothness": cfg.MODEL.DIFFRAS.INV_SMOOTHNESS_SCHED[0],
            "resolution_list": cfg.MODEL.DIFFRAS.RESOLUTIONS,
            "enable_box_sup": enable_box_sup,
            "mask_stride": cfg.MODEL.POLYGON_HEAD.MASK_STRIDE,
            "crop_predicts": cfg.MODEL.BOX_SUP.CROP_PREDICTS,
            "crop_size": cfg.MODEL.BOX_SUP.CROP_SIZE,
            "mask_padding_size": cfg.MODEL.BOX_SUP.MASK_PADDING_SIZE,
            # lmc
            "shape_point_num": cfg.MODEL.POLYGON_HEAD.shape_point_num,  # lmc
            "shape_para": cfg.MODEL.POLYGON_HEAD.shape_para,  # lmc
            "whratio": cfg.MODEL.POLYGON_HEAD.whratio,  # lmc
            "KAN": cfg.MODEL.POLYGON_HEAD.KAN,  # lmc
            "MLP_KAN": cfg.MODEL.POLYGON_HEAD.MLP_KAN,  # lmc
            "merge": cfg.MODEL.POLYGON_HEAD.MERGE,
            "merge_weight": cfg.MODEL.POLYGON_HEAD.MERGE_WEIGHT,
            "fusion_gat": cfg.MODEL.POLYGON_HEAD.FUSION_GAT,
            "decode_gat": cfg.MODEL.POLYGON_HEAD.DECODE_GAT,
            "cross_atten_points": cfg.MODEL.POLYGON_HEAD.CROSS_POINTS,
            "device": cfg.MODEL.DEVICE,
            "Kan_loss": cfg.MODEL.LOSS_KAN_REG,
        }

        ret.update(
            input_shape=input_shape,
        )
        return ret

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def shape_get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_w, valid_ratio_w, valid_ratio_w, valid_ratio_w], -1)
        return valid_ratio

    def build_base_shape(self, batch_size, max_instances, instances, device):
        base_shape = torch.tensor(self.shape_para, dtype=torch.float32, device=device)  # shape: [5]
        base_shape = base_shape[None, None, :].expand(batch_size, max_instances, -1).unsqueeze(2)
        base_shape = base_shape.detach().clone().requires_grad_(base_shape.requires_grad)
        inst_classes_shape = pad_inst_classes(instances, max_instances, self.training, device)
        inst_classes = torch.cat([inst.gt_classes if self.training else inst.pred_classes for inst in instances])
        base_shape = modify_base_shape(base_shape, inst_classes_shape, ratio = self.whratio)
        return base_shape, inst_classes, inst_classes_shape

    def catch_embenddins(self, x, number_levels, batch_size, instances, device):
        masks = []
        pos_embeds = []
        srcs = []

        for l in range(number_levels):
            srcs.append(x[l])
            mask = torch.zeros((batch_size, x[l].shape[-2], x[l].shape[-1]), dtype=torch.bool, device=device)
            masks.append(mask)
            # todo, for non-pooled situation.. actually get the mask.
            f = NestedTensor(x[l], mask)
            pos_embeds.append(self.position_embedding(f))

        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed.weight[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)

        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        number_instances = [len(inst) for inst in instances]
        max_instances = max(number_instances)
        box_preds_xyxy = pad_sequence([
            (inst.proposal_boxes.tensor if self.training else inst.pred_boxes.tensor) / torch.Tensor(2 * inst.image_size[::-1]).to(device)
            for inst in instances], batch_first=True)

        return src_flatten, mask_flatten, lvl_pos_embed_flatten, level_start_index, valid_ratios, spatial_shapes, \
            number_instances, max_instances, box_preds_xyxy

    def forward(self, images, x, instances: List[Instances]):
        mask_wh = torch.as_tensor(x[self.mask_stride_lvl_name].shape[-2:][::-1])
        x = [x[f] for f in self.in_features] # 输入特征图
        device = x[0].device
        mask_wh = mask_wh.to(device=device) # 100，100

        if self.prepool:
            if False:
                input_shapes = [x_.shape[-2:] for x_ in x]
                input_ys = [torch.linspace(-1, 1, s[0], device=device) for s in input_shapes]
                input_xs = [torch.linspace(-1, 1, s[1], device=device) for s in input_shapes]
                input_grid = [torch.stack(torch.meshgrid(y_, x_), dim=-1).unsqueeze(0).repeat(x[0].shape[0], 1, 1, 1)
                              for y_, x_ in zip(input_ys, input_xs)]
                x = [F.grid_sample(x_, grid_) for x_, grid_ in zip(x, input_grid)]
            else:
                # todo, find out how the core reason this works so well.
                aligner = MultiROIPooler(
                    list(itertools.chain.from_iterable([[tuple(x_.shape[-2:])] for x_ in x])),
                    scales=(0.25, 0.125, 0.0625, 0.03125),  # correspongding with scale of P2, P3, P4, and P5
                    sampling_ratio=0,
                    pooler_type="ROIAlignV2",
                    # pooler_type="ROIAlign",
                    assign_to_single_level=False)

                x = aligner(x,
                            [Boxes(torch.Tensor([[0, 0, inst.image_size[1], inst.image_size[0]]]).to(x[0].device)) for
                             inst in instances])

        if self.feature_proj is not None:  # default None # 这个可以改成KAN！
            #x = [self.feature_proj[i](x_.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) for i, x_ in enumerate(x)]
            x = [self.feature_proj[i](x_) for i, x_ in enumerate(x)]

        number_levels = len(x)
        batch_size, feat_dim = x[0].shape[:2]

        # empty instance during inference
        if not self.training:
            no_instances = len(instances[0]) == 0
            if no_instances:
                instances[0].pred_masks = torch.zeros((0, 1, 4, 4), device=device)
                return instances

        # prepare for decoder
        src_flatten, mask_flatten, lvl_pos_embed_flatten, level_start_index, valid_ratios, spatial_shapes, \
            number_instances, max_instances, box_preds_xyxy = self.catch_embenddins(x, number_levels, batch_size, instances, device)

        # points/shape embeddings
        query_embed, tgt = torch.split(self.point_embedding, self.model_dimension, dim=1)
        query_embed = query_embed.unsqueeze(0).unsqueeze(0).expand(batch_size, max_instances, -1, -1)
        tgt = tgt.unsqueeze(0).unsqueeze(0).expand(batch_size, max_instances, -1, -1)

        shape_query_embed, shape_tgt = torch.split(self.shape_embedding, self.model_dimension, dim=1)
        shape_query_embed = shape_query_embed.unsqueeze(0).unsqueeze(0).expand(batch_size, max_instances, -1, -1)
        shape_tgt = shape_tgt.unsqueeze(0).unsqueeze(0).expand(batch_size, max_instances, -1, -1)

        # build base_shape
        base_shape, inst_classes, inst_classes_shape = self.build_base_shape(batch_size, max_instances, instances, device)

        # rescale
        if not self.predict_in_box_space:
            box_preds_xywh = [BoxMode.convert((inst.proposal_boxes.tensor if self.training else inst.pred_boxes.tensor),
                                              BoxMode.XYXY_ABS, BoxMode.XYWH_ABS) for inst in instances] # shape=(B, N, 4)
            padded_box_preds_xywh = pad_sequence(box_preds_xywh, batch_first=True)
            padded_box_xy, padded_box_wh = torch.split(padded_box_preds_xywh.unsqueeze(-2), 2, dim=-1)  # shape=(B, N, 1, 2)
            # 这是角点与长宽，不是中心点与长宽
            pad_img_wh = mask_wh * self.mask_stride  # devide
            # 这一步是将框选内固定的位置，转换成图像中的比例
            padded_box_x, padded_box_y, padded_box_w, padded_box_h = torch.split(padded_box_preds_xywh, 1, dim=-1)
            base_shape[:, :, :, 0] = base_shape[:,:,:,0] * padded_box_w / pad_img_wh[0]
            base_shape[:, :, :, 1] = base_shape[:,:,:,1] * padded_box_h / pad_img_wh[1]
            base_shape[:, :, :, 2] = (base_shape[:,:,:,2] * padded_box_w + padded_box_x) / pad_img_wh[0]
            base_shape[:, :, :, 3] = (base_shape[:,:,:,3] * padded_box_h + padded_box_y) / pad_img_wh[1]

        memory = src_flatten + lvl_pos_embed_flatten
        if self.use_cls_token:
            shape_cls_embed = self.shape_class_embedding_layer(inst_classes_shape).unsqueeze(2).expand(-1, -1, self.shape_point_num, -1)
            point_cls_embed = self.point_class_embedding_layer(inst_classes_shape).unsqueeze(2).expand(-1, -1, self.shape_point_num, -1)

        # decoder shapes
        shape_intermediate, shape_outputs_coords, shape_regularization_loss = self.shape_decode(
            (shape_tgt + shape_cls_embed), base_shape, memory, spatial_shapes, level_start_index, valid_ratios, shape_query_embed, mask_flatten,
            cls_token=None, reference_boxes=box_preds_xyxy, inst_classes_shape = inst_classes_shape)
        shape_outputs_points = []
        shape_outputs_coords_final = shape_outputs_coords[-1]
        for i in range(len(shape_outputs_coords)):
            output_coords = shape_outputs_coords[i]
            shape_outputs_coords[i] = torch.cat([output_coords[j, :number_instances[j]] for j in range(batch_size)])
            output_features = shape_intermediate[i]
            shape_intermediate[i] = torch.cat([output_features[j, :number_instances[j]] for j in range(batch_size)])

        for i in range(len(shape_outputs_coords)):#ss = instances[0].pred_classes
            shape_outputs_points.append(batch_get_rotated_rect_vertices(shape_outputs_coords[i], inst_classes))

        reference_points_real = batch_get_rotated_rect_vertices_points(shape_outputs_coords_final, inst_classes_shape)
        point_intermediate, outputs_coords, point_regularization_loss = self.decoder(
            (tgt + point_cls_embed), reference_points_real, memory, spatial_shapes, level_start_index, valid_ratios, query_embed, mask_flatten,
            cls_token=None, reference_boxes=box_preds_xyxy)
        for i in range(len(outputs_coords)):
            output_coords = outputs_coords[i]
            outputs_coords[i] = torch.cat([output_coords[j, :number_instances[j]] for j in range(batch_size)])
            output_features = point_intermediate[i]
            point_intermediate[i] = torch.cat([output_features[j, :number_instances[j]] for j in range(batch_size)])
        # decoder end

        outputs_coords = shape_outputs_points + outputs_coords
        point_intermediate = shape_intermediate + point_intermediate

        shape_point_weights = None
        if self.merge:
            outputs_coord_f, shape_point_weights = self.fusion(outputs_coords, point_intermediate)
            outputs_coords.append(outputs_coord_f)

        del point_intermediate
        del lvl_pos_embed_flatten
        del mask_flatten
        del memory
        del output_features
        del shape_intermediate
        del query_embed
        del shape_cls_embed
        del shape_query_embed
        del src_flatten
        del tgt
        del shape_tgt
        del shape_outputs_points
        del base_shape
        del point_cls_embed

        if self.training:
            if not self.deep_supervision: # (lmc No)
                outputs_coords = [outputs_coords[-1]]
            # box prediction in BoundaryFormer (lmc NO)
            if self.predict_in_box_space:
                pred_masks = [
                    self.pred_rasterizer(output_coords * float(self.rasterize_at[lid][1].item()) - self.offset,  # to P3
                                         self.rasterize_at[lid][1].item(),
                                         self.rasterize_at[lid][0].item(),
                                         1.0).unsqueeze(1)
                    for lid, output_coords in enumerate(outputs_coords)]
                pred_polygons = [output_coords * float(self.rasterize_at[lid][1].item())
                                 # NOTE: Does minusing offset matter?
                                 for lid, output_coords in
                                 enumerate(outputs_coords)]  # List(Tensor, ...) Tensor shape=(N_inst, Np, 2)
            # clipping strategy in BoxSnake
            elif self.crop_predicts:
                gt_boxes_xyxy = [BoxMode.convert((inst.gt_boxes.tensor), BoxMode.XYXY_ABS, BoxMode.XYWH_ABS) for inst in instances] # shape=(B, N, 4)
                gt_box_xy, gt_box_wh = torch.split(torch.cat(gt_boxes_xyxy, dim=0).unsqueeze(-2), 2, dim=-1)  # shape=(N, 1, 2)
                gt_box_wh = gt_box_wh.clamp(min=1.0)  # after padded sequence, wh may be zero.
                '''
                output_coords =  outputs_coords[0]
                s1 = output_coords * pad_img_wh # 真实坐标
                s2 = output_coords * pad_img_wh - gt_box_xy # 减掉左上角的位置
                s3 = (output_coords * pad_img_wh - gt_box_xy) / gt_box_wh # 变形，映射到框内区域，在框内的相对位置
                s4 = (output_coords * pad_img_wh - gt_box_xy) / gt_box_wh * float(self.crop_size) # 按比例扩大
                s5 = (output_coords * pad_img_wh - gt_box_xy) / gt_box_wh * float(self.crop_size) - self.offset # 偏移量
                print('s')
                '''
                pred_masks = [self.pred_rasterizer((output_coords * pad_img_wh - gt_box_xy) / gt_box_wh * float(self.crop_size) - self.offset,
                                                   self.crop_size,  # w
                                                   self.crop_size,  # h
                                                   1.0).unsqueeze(1).contiguous()  # the output is (N, 1, H, W)
                              for lid, output_coords in enumerate(outputs_coords)]
                # here, we allow the vertices superpass the box. next, we padding the rasterized mask
                pred_masks = [
                    F.pad(pred_mask, tuple(4*[self.mask_padding_size]), mode='constant', value=0.)
                    for pred_mask in pred_masks]
                # pred_polygons lies in the original size
                pred_polygons = [output_coords * pad_img_wh for lid, output_coords in enumerate(outputs_coords)]
                pred_shapes = [torch.cat([shape_output_coords[:,0,0:2] * pad_img_wh, shape_output_coords[:,0,4:] * torch.pi / 2], dim=1)
                               for lid, shape_output_coords in enumerate(shape_outputs_coords)]# original resoluation
                # List(Tensor, ...) Tensor shape=(N_inst, Np, 2)
            # pred in mask scale, like P3, P4
            else:
                pred_masks = [self.pred_rasterizer(output_coords * mask_wh - self.offset,
                                                   mask_wh[0].item(),  # w
                                                   mask_wh[1].item(),  # h
                                                   1.0).unsqueeze(1).contiguous()  # the output is (N, 1, H, W)
                              for lid, output_coords in enumerate(outputs_coords)]
                pred_polygons = [output_coords * pad_img_wh  # NOTE: Does minusing offset matter?
                                 for lid, output_coords in
                                 enumerate(outputs_coords)]  # List(Tensor, ...) Tensor shape=(N_inst, Np, 2)

            if self.enable_box_sup:
                preds = {'mask': pred_masks, 'polygon': pred_polygons, 'shape': pred_shapes}
                pred_level = self.shape_number_layers * ['shape'] + self.point_number_layers * ['point']
                if self.merge:
                    pred_level.append('merge')
                if (not self.MLP_KAN) or (not self.Kan_loss_used):
                    return self.box_supervisor(images, preds, instances, pred_level), shape_point_weights
                else:
                    box_loss = self.box_supervisor(images, preds, instances, pred_level)
                    box_loss.update(
                        {"loss_KAN": (shape_regularization_loss + point_regularization_loss).mean() * self.Kan_loss})
                    return box_loss, shape_point_weights

            return None

        #self.idx_output = 3
        pred_polys_per_image = outputs_coords[self.idx_output].split(number_instances, dim=0) # 最后一个！
        for pred_polys, instance in zip(pred_polys_per_image, instances):
            pred_polys = pred_polys if self.predict_in_box_space else pred_polys * pad_img_wh
            instance.pred_polys = pred_polys

        for i in range(len(outputs_coords)):  # 遍历所有预测头
            pred_polys_per_image = outputs_coords[i].split(number_instances, dim=0)
            for pred_polys, instance in zip(pred_polys_per_image, instances):
                if not self.predict_in_box_space:
                    pred_polys = pred_polys * pad_img_wh
                setattr(instance, f'pred_polys_head{i}', pred_polys)  # 为每个头保存结果
        return instances, shape_point_weights