import os
from detectron2.config import CfgNode as CN

def add_boundaryformer_config(cfg):
    cfg.MODEL.ROI_BOX_HEAD.BBOX_CLS_LOSS_WEIGHT = 1.0

    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    cfg.INPUT.COLOR_AUG_SSD = False
    cfg.INPUT.RB_L = 1.0
    cfg.INPUT.RB_H = 1.0
    cfg.INPUT.RC_L = 1.0
    cfg.INPUT.RC_H = 1.0
    cfg.INPUT.RS_L = 1.0
    cfg.INPUT.RS_H = 1.0
    
    cfg.MODEL.POLYGON_HEAD = CN()
    cfg.MODEL.POLYGON_HEAD.MODEL_DIM = 256
    cfg.MODEL.POLYGON_HEAD.IN_FEATURES = ["p2", "p3", "p4", "p5"]
    cfg.MODEL.POLYGON_HEAD.POLY_INIT = "ellipse"
    cfg.MODEL.POLYGON_HEAD.POLY_NUM_PTS = 64
    cfg.MODEL.POLYGON_HEAD.ITER_REFINE = True
    cfg.MODEL.POLYGON_HEAD.UPSAMPLING = True
    cfg.MODEL.POLYGON_HEAD.UPSAMPLING_BASE_NUM_PTS = 8
    cfg.MODEL.POLYGON_HEAD.POINT_NUM_DEC_LAYERS = 4
    cfg.MODEL.POLYGON_HEAD.SHAPE_NUM_DEC_LAYERS = 4
    cfg.MODEL.POLYGON_HEAD.MERGE = False
    cfg.MODEL.POLYGON_HEAD.MERGE_WEIGHT = 0.1
    cfg.MODEL.POLYGON_HEAD.USE_P2P_ATTN = True
    cfg.MODEL.POLYGON_HEAD.CLS_AGNOSTIC_MASK = True
    cfg.MODEL.POLYGON_HEAD.PRED_WITHIN_BOX = False
    cfg.MODEL.POLYGON_HEAD.PREPOOL = True
    cfg.MODEL.POLYGON_HEAD.MAX_PROPOSALS_PER_IMAGE = 0
    cfg.MODEL.POLYGON_HEAD.DROPOUT = 0.0
    cfg.MODEL.POLYGON_HEAD.DEEP_SUPERVISION = True
    cfg.MODEL.POLYGON_HEAD.COARSE_SEM_SEG_HEAD_NAME = ""
    cfg.MODEL.POLYGON_HEAD.DECODE_GAT = False
    cfg.MODEL.POLYGON_HEAD.FUSION_GAT = False
    cfg.MODEL.LOSS_KAN_REG = 1.0

    cfg.MODEL.POLYGON_HEAD.POLY_LOSS = CN()
    cfg.MODEL.POLYGON_HEAD.POLY_LOSS.NAMES = ["MaskRasterizationLoss"]
    cfg.MODEL.POLYGON_HEAD.POLY_LOSS.WS = [1.0]
    cfg.MODEL.POLYGON_HEAD.POLY_LOSS.TYPE = "dice"
    cfg.MODEL.POLYGON_HEAD.CROSS_POINTS = 4
    
    cfg.MODEL.POLYGON_HEAD.MASK_STRIDE = 8
    cfg.MODEL.POLYGON_HEAD.MASK_STRIDE_SUP = 8
    cfg.MODEL.POLYGON_HEAD.DROPOUT = 0.0

    # lmc
    cfg.MODEL.POLYGON_HEAD.shape_point_num= 8 # lmc
    cfg.MODEL.POLYGON_HEAD.shape_para= [1, 1, 0.5, 0.5, 0.5] # lmc
    cfg.MODEL.POLYGON_HEAD.whratio= 0.2  # lmc
    cfg.MODEL.POLYGON_HEAD.KAN= True  # lmc
    cfg.MODEL.POLYGON_HEAD.MLP_KAN= True  # lmc
    cfg.MODEL.POLYGON_HEAD.USE_CLS_TOKEN = True

    cfg.MODEL.DIFFRAS = CN()
    cfg.MODEL.DIFFRAS.RESOLUTIONS = [64, 64] 
    cfg.MODEL.DIFFRAS.RASTERIZE_WITHIN_UNION = False
    cfg.MODEL.DIFFRAS.USE_RASTERIZED_GT = False
    cfg.MODEL.DIFFRAS.INV_SMOOTHNESS_SCHED = (0.001,) #(0.15, 0.005)
    cfg.MODEL.DIFFRAS.INV_SMOOTHNESS_STEPS = () #(50000.)

    cfg.MODEL.ROI_HEADS.PROPOSAL_ONLY_GT = False
    cfg.MODEL.ROI_HEADS.SIZE_INPUT = False

    cfg.SOLVER.OPTIMIZER = "SGD"
    cfg.COMMENT = "NONE"
    cfg.OUTPUT_PREFIX = "outputs" if os.getenv("DETECTRON2_OUTPUTS") is None else os.getenv("DETECTRON2_OUTPUTS")
    cfg.TRAIN_SET_STR = ""
    cfg.CFG_FILE_STR = "default"
    cfg.OPT_STR = "default"

    cfg.MODEL.BOX_SUP = CN()
    cfg.MODEL.BOX_SUP.ENABLE = False
    cfg.MODEL.BOX_SUP.LOSS_POINTS_SHAPE= False # lmc
    cfg.MODEL.BOX_SUP.LOSS_POINTS_SHAPE_WEIGHT= 0. # lmc
    cfg.MODEL.BOX_SUP.LOSS_POINTS_RELA= False # lmc
    cfg.MODEL.BOX_SUP.LOSS_POINTS_RELA_WEIGHT= 0. # lmc
    cfg.MODEL.BOX_SUP.LOSS_POINTSSHAPE_SUP = False # lmc
    cfg.MODEL.BOX_SUP.LOSS_POINTSSHAPE_SUP_WEIGHT = 0. # lmc 1.0
    cfg.MODEL.BOX_SUP.LOSS_PROJ = False
    cfg.MODEL.BOX_SUP.LOSS_PROJ_DICE_WEIGHT = 1.0
    cfg.MODEL.BOX_SUP.LOSS_PROJ_CE_WEIGHT = 1.0
    cfg.MODEL.BOX_SUP.LOSS_PROJ_TYPE = ["dice"]
    
    cfg.MODEL.BOX_SUP.LOSS_POINTS_PROJ = False
    cfg.MODEL.BOX_SUP.LOSS_POINTS_PROJ_WEIGHT = 0.

    cfg.MODEL.BOX_SUP.LOSS_LOCAL_PAIRWISE = False
    cfg.MODEL.BOX_SUP.LOSS_LOCAL_PAIRWISE_WEIGHT = 1.0
    cfg.MODEL.BOX_SUP.LOSS_PAIRWISE_WARMUP_ITER = 1
    cfg.MODEL.BOX_SUP.LOSS_PAIRWISE_COLD_ITER = 0
    cfg.MODEL.BOX_SUP.LOSS_LOCAL_PAIRWISE_TYPE = "v1" # support v1, v2 (boxinst pairwise loss)
    cfg.MODEL.BOX_SUP.LOCAL_PAIRWISE_KERNEL_SIZE = 3
    cfg.MODEL.BOX_SUP.LOCAL_PAIRWISE_DILATION = 1
    cfg.MODEL.BOX_SUP.LOCAL_PAIRWISE_THR = 0.1
    cfg.MODEL.BOX_SUP.LOCAL_PAIRWISE_SIGMA = 2.0
    
    cfg.MODEL.BOX_SUP.LOSS_AVG_PROJ = False
    cfg.MODEL.BOX_SUP.LOSS_AVG_PROJ_DICE_WEIGHT = 0.0
    cfg.MODEL.BOX_SUP.LOSS_AVG_PROJ_CE_WEIGHT = 0.0

    cfg.MODEL.BOX_SUP.LOSS_GLOBAL_PAIRWISE = False
    cfg.MODEL.BOX_SUP.LOSS_GLOBAL_PAIRWISE_WEIGHT = 1.0
    
    cfg.MODEL.BOX_SUP.CROP_PREDICTS = False
    cfg.MODEL.BOX_SUP.CROP_SIZE = 64
    cfg.MODEL.BOX_SUP.MASK_PADDING_SIZE = 4 # clipping strategy 

    # RoI mask head 
    cfg.MODEL.ROI_MASK_HEAD.MASK_HEAD_WEIGHTS = 1.0
    cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES = True
    cfg.MODEL.BOX_SUP.SEG_WEIGHTS = [1.0, 1.0, 1.0]

    cfg.MODEL_PIXEL_STD_BGR = [57.375, 57.120, 58.395]

    # swin transformer backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SWIN.USE_CHECKPOINT = False
    
    
    # FCOS config
    cfg.MODEL.FCOS = CN()
    # This is the number of foreground classes.
    cfg.MODEL.FCOS.NUM_CLASSES = 80
    cfg.MODEL.FCOS.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
    cfg.MODEL.FCOS.FPN_STRIDES = [8, 16, 32, 64, 128]
    cfg.MODEL.FCOS.PRIOR_PROB = 0.01
    cfg.MODEL.FCOS.INFERENCE_TH_TRAIN = 0.05
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = 0.05
    cfg.MODEL.FCOS.NMS_TH = 0.6
    cfg.MODEL.FCOS.PRE_NMS_TOPK_TRAIN = 1000
    cfg.MODEL.FCOS.PRE_NMS_TOPK_TEST = 1000
    cfg.MODEL.FCOS.POST_NMS_TOPK_TRAIN = 100
    cfg.MODEL.FCOS.POST_NMS_TOPK_TEST = 100
    cfg.MODEL.FCOS.TOP_LEVELS = 2
    cfg.MODEL.FCOS.NORM = "GN"  # Support GN or none
    cfg.MODEL.FCOS.USE_SCALE = True

    # The options for the quality of box prediction
    # It can be "ctrness" (as described in FCOS paper) or "iou"
    # Using "iou" here generally has ~0.4 better AP on COCO
    # Note that for compatibility, we still use the term "ctrness" in the code
    cfg.MODEL.FCOS.BOX_QUALITY = "ctrness"

    # Multiply centerness before threshold
    # This will affect the final performance by about 0.05 AP but save some time
    cfg.MODEL.FCOS.THRESH_WITH_CTR = False

    # Focal loss parameters
    cfg.MODEL.FCOS.LOSS_ALPHA = 0.25
    cfg.MODEL.FCOS.LOSS_GAMMA = 2.0

    # The normalizer of the classification loss
    # The normalizer can be "fg" (normalized by the number of the foreground samples),
    # "moving_fg" (normalized by the MOVING number of the foreground samples),
    # or "all" (normalized by the number of all samples)
    cfg.MODEL.FCOS.LOSS_NORMALIZER_CLS = "fg"
    cfg.MODEL.FCOS.LOSS_WEIGHT_CLS = 1.0

    cfg.MODEL.FCOS.SIZES_OF_INTEREST = [64, 128, 256, 512]
    cfg.MODEL.FCOS.USE_RELU = True
    cfg.MODEL.FCOS.USE_DEFORMABLE = False

    # the number of convolutions used in the cls and bbox tower
    cfg.MODEL.FCOS.NUM_CLS_CONVS = 4
    cfg.MODEL.FCOS.NUM_BOX_CONVS = 4
    cfg.MODEL.FCOS.NUM_SHARE_CONVS = 0
    cfg.MODEL.FCOS.CENTER_SAMPLE = True
    cfg.MODEL.FCOS.POS_RADIUS = 1.5
    cfg.MODEL.FCOS.LOC_LOSS_TYPE = 'giou'
    cfg.MODEL.FCOS.YIELD_PROPOSAL = False
    cfg.MODEL.FCOS.YIELD_BOX_FEATURES = False

    cfg.MODEL.RPN.IOU_THRESHOLDS = [0.3, 0.7]
    cfg.MODEL.RPN.NMS_THRESH = 0.7

    cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.5]
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.01
    cfg.MODEL.ROI_HEADS.CLASS_WEIGHTS = [2.0, 1.0, 2.0]
    cfg.MODEL.BOX_SUP.SEG_WEIGHTS = [2.0, 1.0, 2.0]
    cfg.MODEL.ROI_HEADS_FCOS = CN()
    cfg.MODEL.ROI_HEADS_FCOS.MAX_PROPOSALS = -1
    cfg.MODEL.ROI_HEADS_FCOS.TOPK_PROPOSALS_PER_IM = -1
