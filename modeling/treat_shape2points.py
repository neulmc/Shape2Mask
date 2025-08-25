import torch
import math
from datetime import datetime
import os
import sys
import shutil
'''
def batch_get_rotated_rect_vertices(batch_params: torch.Tensor, cls: torch.Tensor) -> torch.Tensor:
    """
    批量计算旋转矩形的四个顶点坐标
    输入:
        batch_params: [batch_size, 1, 5] 或 [batch_size, 5] 的张量，
                     最后一维顺序为 [w, h, cx, cy, theta(弧度)]
    输出:
        vertices: [batch_size, 4, 2] 张量，每个矩形对应4个顶点的(x,y)坐标
    """
    # 确保输入为 [batch_size, 5] 形状
    if batch_params.dim() == 3 and batch_params.size(1) == 1:
        batch_params = batch_params.squeeze(1)

    # 分割参数
    w = batch_params[:, 0]  # [batch_size]
    h = batch_params[:, 1]  # [batch_size]
    cx = batch_params[:, 2]  # [batch_size]
    cy = batch_params[:, 3]  # [batch_size]
    theta = batch_params[:, 4] * (torch.pi / 2)  # [batch_size]

    # 创建结果容器
    batch_size = batch_params.size(0)
    device = batch_params.device
    max_points = max(4, 4)
    vertices = torch.zeros(batch_size, max_points, 2, device=device)

    # 矩形处理 (cls != 1)
    rect_mask = (cls != 1)  # cls为0或2时生成矩形
    if rect_mask.any():
        half_w = w[rect_mask].unsqueeze(-1) / 2
        half_h = h[rect_mask].unsqueeze(-1) / 2

        local_corners = torch.stack([
            torch.cat([-half_w, -half_h], dim=-1),
            torch.cat([half_w, -half_h], dim=-1),
            torch.cat([half_w, half_h], dim=-1),
            torch.cat([-half_w, half_h], dim=-1)
        ], dim=1)

        cos_theta = torch.cos(theta[rect_mask])
        sin_theta = torch.sin(theta[rect_mask])
        rotation_matrix = torch.stack([
            torch.stack([cos_theta, -sin_theta], dim=-1),
            torch.stack([sin_theta, cos_theta], dim=-1)
        ], dim=-2)

        rect_vertices = torch.bmm(local_corners, rotation_matrix) + \
                        torch.stack([cx[rect_mask], cy[rect_mask]], dim=-1).unsqueeze(1)
        vertices[rect_mask, :4, :] = rect_vertices

    # 椭圆处理 (cls == 1)
    ellipse_mask = (cls == 1)
    if ellipse_mask.any():
        #angles = torch.linspace(0, 2 * math.pi, 4, device=device)
        angles = torch.linspace(0, 2 * math.pi, 5, device=device)
        a = w[ellipse_mask].unsqueeze(-1) / 2
        b = h[ellipse_mask].unsqueeze(-1) / 2

        x = a * torch.cos(angles)
        y = b * torch.sin(angles)
        local_ellipse = torch.stack([x, y], dim=-1)

        cos_theta = torch.cos(theta[ellipse_mask])
        sin_theta = torch.sin(theta[ellipse_mask])
        rotation_matrix = torch.stack([
            torch.stack([cos_theta, -sin_theta], dim=-1),
            torch.stack([sin_theta, cos_theta], dim=-1)
        ], dim=-2)

        ellipse_vertices = torch.bmm(local_ellipse, rotation_matrix) + \
                           torch.stack([cx[ellipse_mask], cy[ellipse_mask]], dim=-1).unsqueeze(1)
        vertices[ellipse_mask, :4, :] = ellipse_vertices[:,:4,:]

    vertices = torch.clamp(vertices, min=0, max=1)
    return vertices  # [batch_size, 4, 2]
'''

def xielv_loss(pred_points, A):
    def is_invalid_point_pair(p_i, p_a, p_b, eps=1.1):
        """
        检查是否存在无效斜率（垂直线或水平线）
        p_i, p_a, p_b: (batch_size, 2)
        Returns: (batch_size,)布尔张量，True表示需要跳过
        """
        # 检查P_i与P_a或P_i与P_b是否x/y坐标接近
        x_close_to_pa = torch.abs(p_i[:, 0] - p_a[:, 0]) < eps
        x_close_to_pb = torch.abs(p_i[:, 0] - p_b[:, 0]) < eps
        y_close_to_pa = torch.abs(p_i[:, 1] - p_a[:, 1]) < eps
        y_close_to_pb = torch.abs(p_i[:, 1] - p_b[:, 1]) < eps
        # 任意一种情况均视为无效
        return x_close_to_pa | x_close_to_pb | y_close_to_pa | y_close_to_pb

    def slope_diff(p_i, p_a, p_b, A, eps = 1e-6):
        """
        计算边点p_i与顶点p_a、p_b的斜率差异
        p_i, p_a, p_b: (batch_size, 2)
        返回: 斜率差异的惩罚（弹性阈值A）
        """
        # 计算斜率差 (避免除零，用微小偏移)
        invalid_mask = is_invalid_point_pair(p_i, p_a, p_b)
        valid_mask = ~invalid_mask
        if valid_mask.sum() == 0:
            return 0.0  # 全部无效

        valid_pi = p_i[valid_mask]
        valid_pa = p_a[valid_mask]
        valid_pb = p_b[valid_mask]

        # 3. 计算斜率差（双重验证）
        slope_pi_pa = (valid_pi[:, 1] - valid_pa[:, 1]) / (valid_pi[:, 0] - valid_pa[:, 0] + eps)
        slope_pi_pb = (valid_pi[:, 1] - valid_pb[:, 1]) / (valid_pi[:, 0] - valid_pb[:, 0] + eps)
        diff = torch.abs(slope_pi_pa - slope_pi_pb)

        # 4. 弹性惩罚
        penalty = torch.relu(diff - A)
        return penalty.mean()

    loss = 0.0
    # 边点1在直线(点0, 点2)上
    loss += slope_diff(pred_points[:, 1], pred_points[:, 0], pred_points[:, 2], A)
    # 边点3在直线(点2, 点4)上
    loss += slope_diff(pred_points[:, 3], pred_points[:, 2], pred_points[:, 4], A)
    # 边点5在直线(点4, 点6)上
    loss += slope_diff(pred_points[:, 5], pred_points[:, 4], pred_points[:, 6], A)
    # 边点7在直线(点6, 点0)上
    loss += slope_diff(pred_points[:, 7], pred_points[:, 6], pred_points[:, 0], A)
    return loss

def structured_loss(pred_points):
    """
    pred_points: Tensor of shape (batch_size, 8, 2)
    lambda_constraint: 约束项的权重
    """
    # 基础损失（假设有真实标签）
    # loss_mse = torch.nn.MSELoss()(pred_points, target_points)

    # 几何约束损失
    loss_constraint = 0.0
    # 提取所有点 (x, y)
    x = pred_points[..., 0]  # (batch_size, 8)
    y = pred_points[..., 1]  # (batch_size, 8)
    # ---- x轴约束 ----
    # 边点1: x0 < x1 < x2
    loss_constraint += torch.mean(torch.relu(x[:, 0] - x[:, 1]))  # x1 >= x0
    loss_constraint += torch.mean(torch.relu(x[:, 1] - x[:, 2]))  # x1 <= x2
    # 边点3: x2 < x3 < x4
    loss_constraint += torch.mean(torch.relu(x[:, 2] - x[:, 3]))
    loss_constraint += torch.mean(torch.relu(x[:, 3] - x[:, 4]))
    # 边点5: x6 < x5 < x4
    loss_constraint += torch.mean(torch.relu(x[:, 5] - x[:, 4]))  # x5 <= x4
    loss_constraint += torch.mean(torch.relu(x[:, 6] - x[:, 5]))  # x5 >= x6
    # 边点7: x0 < x7 < x6
    loss_constraint += torch.mean(torch.relu(x[:, 7] - x[:, 6]))
    loss_constraint += torch.mean(torch.relu(x[:, 0] - x[:, 7]))
    # ---- y轴约束 ----
    # 边点1: y2 < y1 < y0
    loss_constraint += torch.mean(torch.relu(y[:, 1] - y[:, 0]))  # y1 <= y0
    loss_constraint += torch.mean(torch.relu(y[:, 2] - y[:, 1]))  # y1 >= y2
    # 边点3: y2 < y3 < y4
    loss_constraint += torch.mean(torch.relu(y[:, 3] - y[:, 4]))
    loss_constraint += torch.mean(torch.relu(y[:, 2] - y[:, 3]))
    # 边点5: y4 < y5 < y6
    loss_constraint += torch.mean(torch.relu(y[:, 5] - y[:, 6]))
    loss_constraint += torch.mean(torch.relu(y[:, 4] - y[:, 5]))
    # 边点7: y0 < y7 < y6
    loss_constraint += torch.mean(torch.relu(y[:, 0] - y[:, 7]))
    loss_constraint += torch.mean(torch.relu(y[:, 7] - y[:, 6]))
    return loss_constraint

def get_geometry_features(proposals):
    """
    计算归一化的几何特征（基于每个proposal对应的图像尺寸）
    返回: Tensor(sum_N, 4) -> [norm_w, norm_h, norm_area, aspect_ratio]
    """
    geo_features = []
    for x in proposals:
        boxes = x.proposal_boxes.tensor  # (N,4) in (x1,y1,x2,y2) format
        img_w = x.image_size[0]  # 当前proposal所属图像的宽度
        img_h = x.image_size[1]  # 当前proposal所属图像的高度

        # 计算原始几何特征
        w = boxes[:, 2] - boxes[:, 0]  # 宽度 (N,)
        h = boxes[:, 3] - boxes[:, 1]  # 高度 (N,)
        area = w * h  # 面积 (N,)
        aspect_ratio = w / (h + 1e-6)  # 长宽比 (N,)

        # 基于图像尺寸归一化
        norm_w = w / img_w  # 宽度归一化到 [0,1]
        norm_h = h / img_h  # 高度归一化到 [0,1]
        norm_area = area / (img_w * img_h)  # 面积归一化到 [0,1]

        # 拼接特征 (N,4)
        geo_feat = torch.stack([norm_w, norm_h, norm_area, aspect_ratio], dim=1)
        geo_features.append(geo_feat)

    return torch.cat(geo_features, dim=0)  # (sum_N,4)

def batch_get_rotated_rect_vertices(batch_params: torch.Tensor,
                                           cls: torch.Tensor) -> torch.Tensor:
    """
    批量生成旋转形状的边界点（支持矩形和椭圆），每个批次仅处理一个物体。
    规则:
        - cls=0或2: 矩形（8个点：4角点+4边中点）
        - cls=1: 椭圆（8个均匀分布点）
    参数:
        batch_params: [batch_size, 1, 5] 张量，顺序为 [w, h, cx, cy, theta]
        cls: [batch_size] 张量，0/2=矩形，1=椭圆
    返回:
        vertices: [batch_size, 8, 2] 张量
    """
    device = batch_params.device
    batch_size = batch_params.shape[0]

    # 去除多余的维度
    params = batch_params.squeeze(1)  # [batch_size, 5]
    w = params[:, 0]  # [batch_size]
    h = params[:, 1]
    cx = params[:, 2]
    cy = params[:, 3]
    theta = params[:, 4] * (torch.pi / 2)  # 逆时针，不超过pi/2

    # 初始化输出
    vertices = torch.zeros(batch_size, 8, 2, device=device)

    # ===== 矩形处理（cls=0或2） =====
    rect_mask = (cls == 0) | (cls == 2)  # [batch_size]
    if rect_mask.any():
        # 矩形8个点模板（4角点+4边中点）
        rect_template = torch.tensor([
            [-0.5, -0.5], [0.0, -0.5], [0.5, -0.5],  # 上边
            [0.5, 0.0], [0.5, 0.5], [0.0, 0.5],      # 右边和下边
            [-0.5, 0.5], [-0.5, 0.0]                 # 左边
        ], device=device)  # [8, 2]

        # 应用宽高缩放 [batch_size, 8, 2]
        wh = torch.stack([w[rect_mask], h[rect_mask]], dim=-1).unsqueeze(1)  # [n_rect, 1, 2]
        scaled = rect_template * wh

        # 构造旋转矩阵 [n_rect, 2, 2]
        theta_rect = theta[rect_mask].unsqueeze(-1)  # [n_rect, 1]
        cos_t = torch.cos(theta_rect)
        sin_t = torch.sin(theta_rect)
        rot_mat = torch.stack([
            torch.cat([cos_t, -sin_t], dim=-1),
            torch.cat([sin_t, cos_t], dim=-1)
        ], dim=-2)  # [n_rect, 2, 2]

        # 旋转并平移 [n_rect, 8, 2]
        rotated = torch.matmul(scaled, rot_mat)
        center = torch.stack([cx[rect_mask], cy[rect_mask]], dim=-1).unsqueeze(1)  # [n_rect, 1, 2]
        vertices[rect_mask] = rotated + center

    # ===== 椭圆处理（cls=1） =====
    ellipse_mask = (cls == 1)
    if ellipse_mask.any():
        # 生成8个均匀角度
        angles = torch.arange(0, 8, device=device) * (2 * math.pi / 8)  # [8]

        # 椭圆参数方程 [n_ellipse, 8, 2]
        a = w[ellipse_mask].unsqueeze(-1) / 2  # x半轴 [n_ellipse, 1]
        b = h[ellipse_mask].unsqueeze(-1) / 2  # y半轴
        x = a * torch.cos(angles)
        y = b * torch.sin(angles)
        points = torch.stack([x, y], dim=-1)  # [n_ellipse, 8, 2]

        # 旋转矩阵 [n_ellipse, 2, 2]
        theta_ell = theta[ellipse_mask].unsqueeze(-1)  # [n_ellipse, 1]
        cos_t = torch.cos(theta_ell)
        sin_t = torch.sin(theta_ell)
        rot_mat = torch.stack([
            torch.cat([cos_t, -sin_t], dim=-1),
            torch.cat([sin_t, cos_t], dim=-1)
        ], dim=-2)

        # 旋转并平移
        rotated = torch.matmul(points, rot_mat)  # [n_ellipse, 8, 2]
        center = torch.stack([cx[ellipse_mask], cy[ellipse_mask]], dim=-1).unsqueeze(1)  # [n_ellipse, 1, 2]
        vertices[ellipse_mask] = rotated + center

    # 限制坐标范围（假设输入是归一化后的值）
    vertices = torch.clamp(vertices, min=0, max=1)
    return vertices

def batch_get_rotated_rect_vertices_points(batch_params: torch.Tensor,
                                           cls: torch.Tensor) -> torch.Tensor:
    """
    批量生成旋转形状的边界点（支持矩形和椭圆）
    规则:
        - cls=0或2: 矩形（8个点：4角点+4边中点）
        - cls=1: 椭圆（8个均匀分布点）
    参数:
        batch_params: [batch_size, num_objects, 1, 5] 张量，顺序为 [w, h, cx, cy, theta]
        cls: [batch_size, num_objects] 张量，0/2=矩形，1=椭圆
    返回:
        vertices: [batch_size, num_objects, 8, 2] 张量
    """
    device = batch_params.device
    batch_size, num_objects, _, _ = batch_params.shape

    # 去除多余的维度
    params = batch_params.squeeze(2)  # [batch, num, 5]
    w = params[..., 0]  # [batch, num]
    h = params[..., 1]
    cx = params[..., 2]
    cy = params[..., 3]
    theta = params[..., 4] * (torch.pi / 2) # 逆时针，不超过pi/2

    # 初始化输出
    vertices = torch.zeros(batch_size, num_objects, 8, 2, device=device)

    # ===== 矩形处理（cls=0或2） =====
    rect_mask = (cls == 0) | (cls == 2)  # 修改关键判断条件
    if rect_mask.any():
        # 矩形8个点模板（4角点+4边中点）
        rect_template = torch.tensor([
            [-0.5, -0.5], [0.0, -0.5], [0.5, -0.5],  # 上边
            [0.5, 0.0], [0.5, 0.5], [0.0, 0.5],  # 右边和下边
            [-0.5, 0.5], [-0.5, 0.0]  # 左边
        ], device=device)  # [8, 2]

        # 应用宽高缩放 [batch, num, 8, 2]
        wh = torch.stack([w, h], dim=-1).unsqueeze(2)  # [batch, num, 1, 2]
        scaled = rect_template * wh

        # 构造旋转矩阵 [batch, num, 2, 2]
        cos_t = torch.cos(theta).unsqueeze(-1)  # [batch, num, 1]
        sin_t = torch.sin(theta).unsqueeze(-1)
        rot_mat = torch.stack([
            torch.cat([cos_t, -sin_t], dim=-1),
            torch.cat([sin_t, cos_t], dim=-1)
        ], dim=-2)

        # 旋转并平移 [batch, num, 8, 2]
        rotated = torch.matmul(
            scaled.reshape(-1, 8, 2),
            rot_mat.reshape(-1, 2, 2)
        ).view(batch_size, -1, 8, 2)
        vertices[rect_mask] = rotated[rect_mask] + torch.stack([cx, cy], dim=-1)[rect_mask].unsqueeze(1)

    # ===== 椭圆处理（cls=1） =====
    ellipse_mask = (cls == 1)
    if ellipse_mask.any():
        # 生成8个不重复的均匀角度（endpoint=False避免首尾重合）
        angles = torch.arange(0, 8, device=device) * (2 * math.pi / 8)

        # 椭圆参数方程 [n_ellipse, 8, 2]
        a = w[ellipse_mask].unsqueeze(-1) / 2  # x半轴 [n_ellipse, 1]
        b = h[ellipse_mask].unsqueeze(-1) / 2  # y半轴
        x = a * torch.cos(angles)
        y = b * torch.sin(angles)
        points = torch.stack([x, y], dim=-1)  # [n_ellipse, 8, 2]

        # 旋转矩阵 [n_ellipse, 2, 2]
        theta_ell = theta[ellipse_mask].unsqueeze(-1)  # [n_ellipse, 1]
        cos_t = torch.cos(theta_ell)
        sin_t = torch.sin(theta_ell)
        rot_mat = torch.stack([
            torch.cat([cos_t, -sin_t], dim=-1),
            torch.cat([sin_t, cos_t], dim=-1)
        ], dim=-2)

        # 旋转并平移
        rotated = torch.matmul(points, rot_mat)  # [n_ellipse, 8, 2]
        center = torch.stack([cx[ellipse_mask], cy[ellipse_mask]], dim=-1).unsqueeze(1)  # [n_ellipse, 1, 2]
        vertices[ellipse_mask] = rotated + center
    vertices = torch.clamp(vertices, min=0, max=1)
    return vertices


def pad_inst_classes(instances, max_inst_per_batch, training, device):
    """
    处理实例类别，不足时填充1（保持batch维度）
    参数:
        instances: List[Instances] (长度为batch_size)
        max_inst_per_batch: 每个batch允许的最大实例数
        training: 是否训练模式
    返回:
        inst_classes: [batch_size, max_inst_per_batch]
    """
    batch_size = len(instances)

    # 初始化结果张量（全部填充1）
    inst_classes = torch.ones(batch_size, max_inst_per_batch,
                              dtype=torch.long, device=device)

    for i, inst in enumerate(instances):
        # 获取当前实例的真实类别（训练/推理模式）
        if training:
            classes = inst.gt_classes if hasattr(inst, 'gt_classes') else torch.tensor([1], device=device)
        else:
            classes = inst.pred_classes if hasattr(inst, 'pred_classes') else torch.tensor([1], device=device)

        # 填充有效类别（不超过max_inst_per_batch）
        num_valid = min(len(classes), max_inst_per_batch)
        inst_classes[i, :num_valid] = classes[:num_valid]

    return inst_classes


import torch


def modify_base_shape(base_shape: torch.Tensor, inst_classes_shape: torch.Tensor, ratio: float):
    """
    根据类别修改base_shape的特定列
    Args:
        base_shape: [batch, num, 1, 5] 张量
        inst_classes_shape: [batch, num] 类别标签 (0或2)
    Returns:
        修改后的base_shape
    """
    # 创建可写副本（避免原地修改问题）
    modified_shape = base_shape.clone()

    # 获取类别掩码
    mask_class0 = (inst_classes_shape == 0).unsqueeze(-1).unsqueeze(-1)  # [batch, num, 1, 1]
    mask_class2 = (inst_classes_shape == 2).unsqueeze(-1).unsqueeze(-1)

    # 修改第二列（索引1）当类别为0时
    modified_shape[..., 1] = torch.where(
        mask_class0.squeeze(-1),  # 去除最后维度
        base_shape[..., 1] * ratio,
        base_shape[..., 1]
    )

    # 修改第一列（索引0）当类别为2时
    modified_shape[..., 0] = torch.where(
        mask_class2.squeeze(-1),
        base_shape[..., 0] * ratio,
        base_shape[..., 0]
    )

    return modified_shape


def copy_current_file(new_filename):
    # 获取当前执行文件的路径
    current_file = sys.argv[0]
    current_filename = os.path.basename(current_file)
    #new_filename = os.path.splitext(current_filename)[0] + '_copy' + os.path.splitext(current_filename)[1]
    try:
        # 使用shutil.copy2()保留文件元数据
        shutil.copy2(current_file, new_filename)
    except Exception as e:
        print(f'文件复制失败: {e}')


import numpy as np
from scipy.optimize import leastsq
from scipy.interpolate import splprep, splev


def interpolate_ellipse_points(points,  n_interpolated=8):
    """
    保持旋转顺序的角度插值（原始点+插值点合并输出）

    参数:
        points: 输入点坐标，形状为[N,2]的numpy数组
        n_interpolated: 需要插入的点数量（默认8，总点数=N+n_interpolated）
        visualize: 是否可视化结果（默认False）

    返回:
        合并后的有序点坐标，形状为[N+n_interpolated, 2]的numpy数组
    """
    # 输入检查
    if not isinstance(points, np.ndarray) or points.shape[1] != 2:
        raise ValueError("输入必须是Nx2的numpy数组")

    # 计算中心点
    center = np.mean(points, axis=0)
    translated = points - center

    # 计算角度和半径（按旋转顺序排序）
    angles = np.arctan2(translated[:, 1], translated[:, 0])  # [-π, π]
    radii = np.linalg.norm(translated, axis=1)

    # 按角度排序原始点（确保顺时针/逆时针顺序）
    sort_idx = np.argsort(angles)
    sorted_angles = angles[sort_idx]
    sorted_radii = radii[sort_idx]
    sorted_points = points[sort_idx]

    # 在每两个原始点之间插入新点
    new_points = []
    for i in range(len(sorted_angles)):
        # 当前点和下一个点（循环处理）
        ang1, rad1, pt1 = sorted_angles[i], sorted_radii[i], sorted_points[i]
        ang2 = sorted_angles[(i + 1) % len(sorted_angles)]
        rad2 = sorted_radii[(i + 1) % len(sorted_angles)]

        # 处理角度跨越2π的情况（如从π跳到-π）
        if ang2 < ang1:
            ang2 += 2 * np.pi

        # 在两个角度之间均匀插值
        interp_angles = np.linspace(ang1, ang2, n_interpolated // len(points) + 2)[1:-1]
        interp_radii = np.interp(interp_angles, [ang1, ang2], [rad1, rad2])

        # 转换回直角坐标
        interp_x = center[0] + interp_radii * np.cos(interp_angles)
        interp_y = center[1] + interp_radii * np.sin(interp_angles)
        new_points.extend(zip(interp_x, interp_y))

    # 合并原始点和插值点
    combined = np.vstack([sorted_points, new_points])

    # 按角度重新排序所有点
    final_angles = np.arctan2(combined[:, 1] - center[1], combined[:, 0] - center[0])
    result = combined[np.argsort(final_angles)]
    return result