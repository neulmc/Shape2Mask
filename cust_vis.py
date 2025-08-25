from detectron2.evaluation import COCOEvaluator
from detectron2.utils.visualizer import Visualizer, ColorMode, _KEYPOINT_THRESHOLD
import os
import cv2
from detectron2.data import MetadataCatalog
from copy import deepcopy
import torch
import numpy as np
from detectron2.structures import Boxes, Instances, PolygonMasks, BitMasks
from pycocotools import mask as coco_mask
import pickle
import matplotlib.pyplot as plt
import copy


class CustomVisualizer(Visualizer):
    def draw_and_connect_keypoints(self, keypoints):
        """
        重写关键点绘制方法，支持4维关键点（x,y,v,cls）
        """
        #keypoints = keypoints[:, :, :3]  # 只取前3维用于默认绘制
        #super().draw_and_connect_keypoints(keypoints)  # 调用父类方法绘制基础关键点

        # 额外按类别绘制（可选）
        if hasattr(self.metadata, "keypoint_colors"):
            for kpt in keypoints:
                x, y, v, cls = kpt[0], kpt[1], kpt[2], kpt[3]
                color = self.metadata.keypoint_colors[int(cls)]
                # 2. 绘制所有相邻关键点的连线（简单直线）
                for i in range(len(keypoints) - 1):
                    x1, y1 = keypoints[i][0], keypoints[i][1]
                    x2, y2 = keypoints[i + 1][0], keypoints[i + 1][1]
                    self.output.ax.plot([x1, x2], [y1, y2], color=np.array(color) / 255, linewidth=1.5, alpha=0.8, zorder=1)
                x1, y1 = keypoints[-1][0], keypoints[-1][1]
                x2, y2 = keypoints[0][0], keypoints[0][1]
                self.output.ax.plot([x1, x2], [y1, y2], color=np.array(color) / 255, linewidth=1.5, alpha=0.8, zorder=1)
                if v > 0:
                    # 红色外圈
                    self.output.ax.add_patch(
                        plt.Circle((x, y), 4, fill=False, color=(0,0,0), linewidth=2, zorder=2)
                    )
                    self.output.ax.add_patch(
                        plt.Circle((x, y), radius=3, fill=True, color=np.array(color) / 255, zorder=3)
                    )
        self.output.ax.autoscale_view()  # 自动缩放视图
        self.output.ax.axis('off')  # 隐藏坐标轴
        self.output.ax.margins(0)  # 移除所有边距

class CustomCOCOEvaluator(COCOEvaluator):
    def __init__(self, dataset_name, output_dir=None, save_visualization=False,
                 score_threshold=0.1, max_instances=300, num_layers=1, atten_merge=False,
                 matrix_dir='No'):   # 最大实例数量（None表示不限制）):
        super().__init__(dataset_name, output_dir=output_dir)
        self.save_visualization = save_visualization
        self.visualization_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(self.visualization_dir, exist_ok=True)
        self.metadata = MetadataCatalog.get(dataset_name)
        score_threshold = 0.5 # 改，将其阈值限制为0.5以下
        self.score_threshold = score_threshold
        self.max_instances = max_instances  # 新增：最大实例数量限制
        self.num_layers = num_layers
        self.atten_merge = atten_merge
        self.matrix_dir = matrix_dir

    def _create_gt_instances(self, annotations, image_shape):
        """
        将COCO格式的GT标注转换为Instances对象
        """
        from detectron2.structures import Boxes, Instances

        instances = Instances(image_shape)
        height, width = image_shape
        boxes = [ann["bbox"] for ann in annotations]  # COCO格式是[x,y,w,h]
        boxes = np.array([[x, y, x + w, y + h] for x, y, w, h in boxes])  # 转为[x1,y1,x2,y2]

        instances.pred_boxes = Boxes(boxes)
        instances.pred_classes = torch.tensor([ann["category_id"] for ann in annotations], dtype=torch.int64)

        # 如果有mask（语义分割任务）
        if annotations and "segmentation" in annotations[0]:
            bitmask_list = []
            for ann in annotations:
                segm = ann["segmentation"]
                if isinstance(segm, list):  # Polygon格式 → 转RLE → 转Bitmask
                    rle = coco_mask.frPyObjects(segm, height, width)
                    mask = coco_mask.decode(rle)  # 返回uint8的HxW数组
                elif isinstance(segm, dict):  # RLE格式
                    mask = coco_mask.decode(segm)
                elif isinstance(segm, np.ndarray):  # 已经是二值数组
                    mask = segm.astype(np.uint8)
                else:
                    raise ValueError(f"Unsupported segmentation type: {type(segm)}")

                # 确保是2维的0/1数组
                if mask.ndim == 3:
                    mask = mask[:, :, 0]  # 取第一个通道（如果有多个）
                bitmask_list.append(mask)

            # 堆叠为[N,H,W]的tensor
            bitmask_array = np.stack(bitmask_list)
            instances.pred_masks = BitMasks(torch.from_numpy(bitmask_array))
        return instances

    def convert_to_instances(self, matrix_bbox, matrix_mask, image_size):
        """
        将MMDetection格式的输出转换为Detectron2的Instances对象

        参数:
            matrix_bbox: 列表，包含3个numpy数组，每个数组形状为[N,5] (x1,y1,x2,y2,score)
            matrix_mask: 列表，包含3个numpy数组，每个数组形状为[N,1024,1024]的bitmap
            image_size: 元组，图像的(height, width)

        返回:
            Instances对象
        """
        height, width = image_size
        instances = Instances((height, width))

        # 合并所有类别的bbox和score
        all_boxes = []
        all_scores = []
        all_classes = []
        all_masks = []

        for class_id, (bboxes, masks) in enumerate(zip(matrix_bbox, matrix_mask)):
            if len(bboxes) > 0:
                all_boxes.append(bboxes[:, :4])  # 取前4列作为bbox坐标
                all_scores.append(bboxes[:, 4])  # 第5列是score
                all_classes.append(np.full(len(bboxes), class_id))
                all_masks += masks

        if len(all_boxes) == 0:
            return instances  # 如果没有检测结果，返回空实例

        # 合并所有结果
        boxes = np.concatenate(all_boxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        classes = np.concatenate(all_classes, axis=0)
        masks = np.array(all_masks)

        # 转换为torch tensor并设置到Instances对象
        instances.pred_boxes = torch.from_numpy(boxes).float()
        instances.scores = torch.from_numpy(scores).float()
        instances.pred_classes = torch.from_numpy(classes).long()

        if masks is not None:
            instances.pred_masks = torch.from_numpy(masks).bool()

        return instances

    def process(self, inputs, outputs):
        super().process(inputs, outputs)
        # lmc go
        dirs_ = os.listdir(self.visualization_dir)
        iters = 0
        for dir_ in dirs_:
            if iters < int(dir_):
                iters =int(dir_)
        save_dir_now = self.visualization_dir + '/' + str(iters)
        if not os.path.exists(save_dir_now):
            os.makedirs(save_dir_now)
        if len(os.listdir(self.visualization_dir + '/' + str(iters))) == int(30 * (self.num_layers +1)):
            iters += 1
        save_dir_now = self.visualization_dir + '/' + str(iters)
        if not os.path.exists(save_dir_now):
            os.makedirs(save_dir_now)
        if self.save_visualization:
            color_map = [
                (160, 30, 140, 180),  # 深邃紫红 (降低亮度，增加紫调)
                (0, 120, 255, 180),  # 保持原亮蓝色 (Pantone 285C)
                (255, 180, 40, 180),  # 琥珀黄 (降低亮度，减少刺眼感)
            ]
            #self.metadata.thing_colors = color_map
            self.metadata = MetadataCatalog.get("metal_img").set(thing_classes=["Hor", "Ver", "Lon"])
            self.metadata.thing_colors = color_map

            for input, output in zip(inputs, outputs):
                img = cv2.imread(input["file_name"])
                visualizer = Visualizer(img[:, :, ::-1], metadata=self.metadata, scale=1.0, instance_mode=ColorMode.SEGMENTATION)

                if "annotations" in input:
                    visualizer_gt = Visualizer(img[:, :, ::-1], metadata=self.metadata, scale=1.0, instance_mode=ColorMode.SEGMENTATION)
                    gt_annotations = input["annotations"]
                    # 将GT标注转换为Instances对象
                    gt_instances = self._create_gt_instances(gt_annotations, img.shape[:2])
                    # 绘制GT（使用不同颜色或样式区分）
                    # 获取类别为0的实例掩码
                    #keep_mask = gt_instances.pred_classes == 2
                    #gt_instances = gt_instances[keep_mask]

                    #boxes = gt_instances.pred_boxes.tensor  # [N, 4] 格式 (x1, y1, x2, y2)
                    #widths = boxes[:, 2] - boxes[:, 0]  # 计算宽度 (x2 - x1)
                    #heights = boxes[:, 3] - boxes[:, 1]  # 计算高度 (y2 - y1)
                    #keep_mask = widths <= heights
                    #gt_instances = gt_instances[keep_mask]

                    vis_gt = visualizer_gt.draw_instance_predictions(gt_instances, jittering=False)
                    img_name = os.path.basename(input["file_name"])
                    save_path = os.path.join(save_dir_now, img_name+'.gt.png')
                    cv2.imwrite(save_path, vis_gt.get_image()[:, :, ::-1])

                # 可视化预测结果
                if self.matrix_dir != 'No':
                    visualizer_matrix = Visualizer(img[:, :, ::-1], metadata=self.metadata, scale=1.0, instance_mode=ColorMode.SEGMENTATION)
                    matrix_mask = pickle.load(open(self.matrix_dir + '/' + input["file_name"].split('\\')[-1] + '_mask.pkl','rb'))
                    matrix_bbox = pickle.load(open(self.matrix_dir + '/' + input["file_name"].split('\\')[-1] + '_bbox.pkl','rb'))
                    matrix_instances = self.convert_to_instances(matrix_bbox, matrix_mask, img.shape[:2])
                    keep_mask = matrix_instances.scores > self.score_threshold
                    filtered_matrix_instances = deepcopy(matrix_instances[keep_mask])
                    if hasattr(filtered_matrix_instances, 'scores'):
                        filtered_matrix_instances.remove('scores')  # 直接删除 scores 字段
                    vis_gt = visualizer_matrix.draw_instance_predictions(filtered_matrix_instances, jittering=False)
                    img_name = os.path.basename(input["file_name"])
                    save_path = os.path.join(save_dir_now, img_name+'.matrix.png')
                    cv2.imwrite(save_path, vis_gt.get_image()[:, :, ::-1])
                    return

                instances = output["instances"].to("cpu")
                keep_mask = instances.scores > self.score_threshold
                filtered_instances = deepcopy(instances[keep_mask])
                if self.max_instances is not None and len(filtered_instances) > self.max_instances:
                    # 按分数降序排序，取前 max_instances 个
                    scores = filtered_instances.scores
                    _, top_indices = scores.topk(self.max_instances)
                    filtered_instances = filtered_instances[top_indices]

                if hasattr(filtered_instances, 'scores'):
                    filtered_instances.remove('scores')  # 直接删除 scores 字段
                # 获取类别为0的实例掩码
                #keep_mask = filtered_instances.pred_classes == 2
                #filtered_instances = filtered_instances[keep_mask]

                vis = visualizer.draw_instance_predictions(filtered_instances, jittering=False)
                # 保存可视化结果
                img_name = os.path.basename(input["file_name"])
                if self.atten_merge:
                    #attn = torch.mean(instances.shape_point_weights, dim = 0).detach().numpy()
                    with torch.no_grad():  # 确保不保留计算图
                        attn = torch.mean(instances.shape_point_weights.float(), dim=0).cpu().numpy()
                    #del attn  # 显式释放（可选）
                save_path = os.path.join(save_dir_now, img_name)
                cv2.imwrite(save_path, vis.get_image()[:, :, ::-1])
                # 保存中间可视化结果
                for ii in range(self.num_layers):
                    if hasattr(filtered_instances, 'pred_masks_head' +str(ii)):
                        visualizer = CustomVisualizer(img[:, :, ::-1], metadata=self.metadata, scale=1.0, instance_mode=ColorMode.SEGMENTATION)
                        filtered_instances.pred_masks = getattr(filtered_instances, 'pred_masks_head' + str(ii))
                        polygons = getattr(filtered_instances, 'pred_polys_head' + str(ii))
                        # Convert to keypoints format [N, K, 3]
                        class_ids = filtered_instances.pred_classes.numpy()  # 形状 [N]，每个元素是0/1/2
                        keypoints = []
                        for poly, cls_id in zip(polygons, class_ids):
                            kpts = np.hstack([
                                poly,  # 坐标 [K, 2]
                                np.ones((len(poly), 1)),  # 可见性=1 [K, 1]
                                np.full((len(poly), 1), cls_id)  # 类别ID [K, 1]
                            ])  # 合并后形状 [K, 4]
                            keypoints.append(kpts)
                        # Assign to filtered_instances
                        keypoints = np.array(keypoints)
                        if len(keypoints) == 0:
                            continue
                        keypoints[:,:,:2] = keypoints[:,:,:2]*2
                        filtered_instances.pred_keypoints = keypoints
                        #if filtered_instances.has("pred_boxes"):
                        #    filtered_instances.remove("pred_boxes")   # 去掉边界框

                        self.metadata.keypoint_names = ["kp1", "kp2", "kp3"]  # 你的关键点类别名称
                        color_map = [
                            (200, 50, 255),  # 深邃紫红 (降低亮度，增加紫调)
                            (50, 200, 255),  # 保持原亮蓝色 (Pantone 285C)
                            (255,220,80),  # 琥珀黄 (降低亮度，减少刺眼感)
                        ]
                        self.metadata.keypoint_colors = color_map  # 每个关键点的RGB颜色
                        self.metadata.keypoint_connection_rules = [
                            ("kp1", "kp1", (255, 0, 0)),  # 连接kp1和kp2，用红色线
                            ("kp2", "kp2", (0, 255, 0)),  # 连接kp2和kp3，用绿色线
                        ]

                        vis = visualizer.draw_instance_predictions(filtered_instances, jittering=False)
                        # 保存可视化结果
                        img_name = os.path.basename(input["file_name"])
                        if self.atten_merge:
                            save_path = os.path.join(save_dir_now,
                                                     img_name.split('.bmp')[0] + '_pred' + str(ii) + '_atten'+ str(attn[ii]) + '.bmp')
                        else:
                            save_path = os.path.join(save_dir_now,
                                                     img_name.split('.bmp')[0] + '_pred' + str(ii) + '.bmp')
                        cv2.imwrite(save_path, vis.get_image()[:, :, ::-1])