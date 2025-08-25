import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv


class GeometricGAT(nn.Module):
    def __init__(self, fea_dim, hidden_dim, out_dim, num_points, device, heads=4):
        super().__init__()
        self.num_points = num_points
        #self.gat1 = GATConv(fea_dim + 2, int(hidden_dim/heads/4), heads=heads, edge_dim=3)
        #self.gat2 = GATConv(int(hidden_dim/4), out_dim, edge_dim=3)
        self.gat1 = GATConv(fea_dim + 2, int(hidden_dim/heads), heads=heads, edge_dim=3)
        self.gat2 = GATConv(hidden_dim, out_dim, edge_dim=3)
        self.device = device

        # 固定环形邻接矩阵（初始化时生成）
        self.adj = self.build_ring_adjacency(num_points).to(self.device)
        self.edge_index = self.adj.nonzero(as_tuple=False).t()  # [2, num_edges]
        #self.register_buffer('adj', self.build_ring_adjacency(num_points))
        #self.register_buffer('edge_index', self.adj.nonzero(as_tuple=False).t())

    def build_ring_adjacency(self, num_points):
        """构建环形邻接矩阵"""
        adj = torch.zeros(num_points, num_points, dtype=torch.float)
        for i in range(num_points):
            adj[i, (i - 1) % num_points] = 1  # 前驱节点
            adj[i, (i + 1) % num_points] = 1  # 后继节点
        return adj  # [num_points, num_points]

    def forward(self, x, coords):
        # x: [batch, num_points, fea_dim]
        # coords: [batch, num_points, 2]
        batch_size = x.shape[0]
        x = torch.cat([x, coords], dim=-1)  # [batch, num_points, fea_dim+2]

        # 批量计算边特征（边索引固定）
        edge_attr = self.compute_edge_attr_batch(coords)  # [batch, num_edges, 3]

        # 构建批量图
        data_list = [
            Data(x=x[i], edge_index=self.edge_index, edge_attr=edge_attr[i])
            for i in range(batch_size)
        ]
        batch_data = Batch.from_data_list(data_list)

        # GAT处理
        h = self.gat1(batch_data.x, batch_data.edge_index, edge_attr=batch_data.edge_attr)
        h = torch.relu(h)
        h = self.gat2(h, batch_data.edge_index, edge_attr=batch_data.edge_attr)

        return h.view(batch_size, self.num_points, -1)

    def compute_edge_attr_batch(self, coords):
        """动态计算边特征（基于坐标）"""
        batch_size = coords.shape[0]
        src, dst = self.edge_index[0], self.edge_index[1]  # [num_edges]

        # 扩展为批量处理
        src_batch = src.unsqueeze(0).expand(batch_size, -1)  # [batch, num_edges]
        dst_batch = dst.unsqueeze(0).expand(batch_size, -1)  # [batch, num_edges]

        # 计算相对坐标和距离
        delta = torch.gather(coords, 1, dst_batch.unsqueeze(-1).expand(-1, -1, 2)) - \
                torch.gather(coords, 1, src_batch.unsqueeze(-1).expand(-1, -1, 2))  # [batch, num_edges, 2]
        distance = torch.norm(delta, dim=-1, keepdim=True)  # [batch, num_edges, 1]
        edge_attr = torch.cat([delta, distance], dim=-1)  # [batch, num_edges, 3]

        return edge_attr