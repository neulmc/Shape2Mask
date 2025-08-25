import os
import torch
import itertools
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader, DatasetCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    SemSegEvaluator,
    verify_results,
)
from modeling.new_rcnn import NewGeneralizedRCNN
from detectron2.modeling.meta_arch import META_ARCH_REGISTRY
from detectron2.solver import get_default_optimizer_params
from detectron2.solver.build import maybe_add_gradient_clipping
from modeling import add_boundaryformer_config
from modeling.data import BoxSnakeDatasetMapper
from detectron2.data.datasets import register_coco_instances
from cust_vis import CustomCOCOEvaluator
import shutil
from modeling.treat_shape2points import copy_current_file
from detectron2.engine import HookBase

# Environment settings
outname = 'Box_supervision_Shape2Mask'  ## dir-name
cuda_id = "0" # gpu_num
os.environ["OMP_NUM_THREADS"] = "1"  # Turn off OpenMP multithreading
os.environ["MKL_NUM_THREADS"] = "1"  # Turn off MKL multithreading

# Runtime configuration
config_file = 'configs/metal-InstanceSegmentation/BoxSnake_RCNN/shape2point_R_50_FPN_1x.yaml'
save_visualization = False
seed = 42
print_iter = 20

# Network structural hyperparameters
cfg_MLP_KAN = True    # KAN enable
cfg_DECODE_GAT = True # GAT enable
cfg_MERGE = True      # Fusion enable
cfg_shape_layer = 2   # num of shape-KAN module
cfg_point_layer = 2   # num of vertex-KAN module

# Loss function hyperparameters
cfg_SUP_PROJ = True            # Sup-loss enable
cfg_SUP_PROJ_WEIGHT = 1.0
cfg_POINTS_SHAPE = True        # PriorS-loss enable
cfg_POINTS_SHAPE_WEIGHT = 0.1
cfg_POINTS_RELA = True         # PriorR-loss enable
cfg_POINTS_RELA_WEIGHT = 0.1
cfg_LOCAL_PAIRWISE = True      # Image-loss(pair) enable
cfg_LOCAL_PAIRWISE_WEIGHT = 0.5
cfg_GLOBAL_PAIRWISE = True     # Image-loss(global) enable
cfg_GLOBAL_PAIRWISE_WEIGHT = 0.5
cfg_KAN_REG = 1e-4             # The regularization of KAN

# Other hyperparameters (default)
cfg_MODEL_DIM = 256
fpn_OUT_CHANNELS = 256

# prefix code
torch.cuda.empty_cache()  # Clean up video memory shards
DatasetCatalog.clear()  # Only removes dataset names and loading functions from memory
MetadataCatalog.clear()  # Only removes metadata from memory

META_ARCH_REGISTRY._obj_map.clear()
META_ARCH_REGISTRY.register(NewGeneralizedRCNN)
register_coco_instances("metal_test", {}, "datasets/metal/annotations/test.json", "datasets/metal/test")
register_coco_instances("metal_train", {}, "datasets/metal/annotations/train.json", "datasets/metal/train")
MetadataCatalog.get("metal_test").thing_classes = ["heng", "dian", "shu"] # mapping ["Hor", "Ver", "Lon"]
MetadataCatalog.get("metal_train").thing_classes = ["heng", "dian", "shu"]

def trans_cfg(cfg):
    cfg.MODEL.POLYGON_HEAD.MLP_KAN = cfg_MLP_KAN
    cfg.MODEL.POLYGON_HEAD.DECODE_GAT = cfg_DECODE_GAT
    cfg.MODEL.POLYGON_HEAD.MERGE = cfg_MERGE

    cfg.MODEL.BOX_SUP.LOSS_POINTS_SHAPE = cfg_POINTS_SHAPE
    cfg.MODEL.BOX_SUP.LOSS_POINTS_SHAPE_WEIGHT = cfg_POINTS_SHAPE_WEIGHT
    cfg.MODEL.BOX_SUP.LOSS_POINTS_RELA = cfg_POINTS_RELA
    cfg.MODEL.BOX_SUP.LOSS_POINTS_RELA_WEIGHT = cfg_POINTS_RELA_WEIGHT
    cfg.MODEL.BOX_SUP.LOSS_POINTS_PROJ = cfg_SUP_PROJ
    cfg.MODEL.BOX_SUP.LOSS_POINTS_PROJ_WEIGHT = cfg_SUP_PROJ_WEIGHT
    cfg.MODEL.BOX_SUP.LOSS_LOCAL_PAIRWISE = cfg_LOCAL_PAIRWISE
    cfg.MODEL.BOX_SUP.LOSS_LOCAL_PAIRWISE_WEIGHT = cfg_LOCAL_PAIRWISE_WEIGHT
    cfg.MODEL.BOX_SUP.LOSS_GLOBAL_PAIRWISE = cfg_GLOBAL_PAIRWISE_WEIGHT
    cfg.MODEL.BOX_SUP.LOSS_GLOBAL_PAIRWISE_WEIGHT = cfg_GLOBAL_PAIRWISE_WEIGHT
    cfg.MODEL.LOSS_KAN_REG = cfg_KAN_REG

    cfg.MODEL.POLYGON_HEAD.MODEL_DIM = cfg_MODEL_DIM
    cfg.MODEL.FPN.OUT_CHANNELS = fpn_OUT_CHANNELS

    cfg.SOLVER.REFERENCE_WORLD_SIZE = 1
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.MODEL.DEVICE = "cuda:" + cuda_id
    cfg.SOLVER.IMS_PER_BATCH = False
    cfg.DATASETS.TRAIN = ("metal_train",)
    cfg.DATASETS.TEST = ("metal_test",)

    cfg.MODEL.POLYGON_HEAD.SHAPE_NUM_DEC_LAYERS = cfg_shape_layer
    cfg.MODEL.POLYGON_HEAD.POINT_NUM_DEC_LAYERS = cfg_point_layer

class EmptyCacheHook(HookBase):
    def after_step(self):
        if self.trainer.iter % 100 == 0:
            torch.cuda.empty_cache()

class Trainer(DefaultTrainer):
    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, EmptyCacheHook())
        return hooks

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if evaluator_type == "coco":
            # return COCOEvaluator(dataset_name, output_dir=output_folder)
            num_layers = int(cfg.MODEL.POLYGON_HEAD.SHAPE_NUM_DEC_LAYERS) + int(
                cfg.MODEL.POLYGON_HEAD.POINT_NUM_DEC_LAYERS)
            atten_merge = cfg.MODEL.POLYGON_HEAD.MERGE
            return CustomCOCOEvaluator(dataset_name, output_dir=output_folder, save_visualization=save_visualization,
                                       num_layers=num_layers, atten_merge=atten_merge)
        if evaluator_type == "sem_seg":
            return SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        if evaluator_type == "cityscapes_instance":
            assert (
                    torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                    torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = None
        if cfg.MODEL.BOX_SUP.ENABLE:
            mapper = BoxSnakeDatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_optimizer(cls, cfg, model):
        overrides = {}
        if cfg.MODEL.BACKBONE.NAME == "build_swin_fpn_backbone":
            overrides.update({
                "absolute_pos_embed": {"lr": cfg.SOLVER.BASE_LR, "weight_decay": 0.0},
                "relative_position_bias_table": {"lr": cfg.SOLVER.BASE_LR, "weight_decay": 0.0},
            })

        params = get_default_optimizer_params(
            model,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
            overrides=overrides
        )

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                    cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                    and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                    and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_gradient_clipping(cfg, torch.optim.SGD)(
                params,
                cfg.SOLVER.BASE_LR,
                momentum=cfg.SOLVER.MOMENTUM,
                nesterov=cfg.SOLVER.NESTEROV,
            )
        elif (optimizer_type == "ADAMW" or optimizer_type == "ADAM") and (
                cfg.MODEL.BACKBONE.NAME != "build_swin_fpn_backbone"):
            optimizer = maybe_add_gradient_clipping(cfg, torch.optim.AdamW)(
                params,
                cfg.SOLVER.BASE_LR
            )  # boundary former optimizer
        elif (optimizer_type == "ADAMW" or optimizer_type == "ADAM") and (
                cfg.MODEL.BACKBONE.NAME == "build_swin_fpn_backbone"):
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR, betas=(0.9, 0.999),  # following mask2former
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        else:
            NotImplementedError(f"no optimizer type {optimizer_type}")

        return optimizer

def setup(args):
    cfg = get_cfg()
    add_boundaryformer_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    trans_cfg(cfg)

    cfg.TRAIN_SET_STR = "+".join(cfg.DATASETS.TRAIN)
    if args.config_file:
        # we also want the enclosing directory.
        dir_name = os.path.basename(os.path.dirname(args.config_file))
        base_name = os.path.basename(args.config_file)

        cfg.CFG_FILE_STR, _ = os.path.splitext(base_name)
        cfg.CFG_FILE_STR = os.path.join(dir_name, base_name)

    IGNORE_KEYS = ["MODEL.WEIGHTS", "SOLVER.IMS_PER_BATCH"]
    if args.opts:
        opt_idx = 0
        kvs = []
        while opt_idx < len(args.opts):
            key, value = args.opts[opt_idx:(opt_idx + 2)]
            if key in IGNORE_KEYS:
                opt_idx += 2
                continue

            # no spaces.
            value = value.replace(" ", "_")

            kvs.append("{0}#{1}".format(key, value))
            opt_idx += 2

        cfg.OPT_STR = "+".join(kvs)

    # compute the train output
    cfg.OUTPUT_DIR = os.path.join(
        cfg.OUTPUT_PREFIX, "train", cfg.TRAIN_SET_STR, cfg.CFG_FILE_STR, cfg.OPT_STR
    )
    cfg["OUTPUT_DIR"] = cfg["OUTPUT_DIR"].replace('default', outname)
    #cfg.freeze()
    default_setup(cfg, args)
    cfg.GLOBAL = None
    cfg.TEST.AUG = None

    cfg.freeze()
    return cfg


def main(args):
    cfg = setup(args)
    cfg["OUTPUT_DIR"] = cfg["OUTPUT_DIR"].replace('default', outname)
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res
    if not os.path.exists(cfg["OUTPUT_DIR"]):
        os.makedirs(cfg["OUTPUT_DIR"])
    copy_current_file(cfg["OUTPUT_DIR"] + '/train_net.py')
    shutil.copy(config_file, cfg["OUTPUT_DIR"] + '/config.yaml')

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_id
    torch.cuda.set_device(int(cuda_id))
    torch.multiprocessing.set_start_method('spawn', force=True)

    args = default_argument_parser().parse_args()
    args.dist_url = "disabled"
    args.config_file = config_file
    #args.resume = True
    #args.eval_only = True
    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )