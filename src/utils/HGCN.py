import torch
import torch.nn as nn
import torch.nn.functional as F

class HGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_agents, num_groups, num_layers):
        super(HGCN, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_agents = num_agents
        self.num_groups = num_groups
        self.num_layers = num_layers
        # print(f"in_dim: {self.in_dim}")

        self.layers = nn.ModuleList()
        self.layers.append(HGCNLayer(in_dim, hidden_dim, num_groups))
        for _ in range(num_layers - 2):
            self.layers.append(HGCNLayer(hidden_dim, hidden_dim, num_groups))
        self.layers.append(HGCNLayer(hidden_dim, out_dim, num_groups))

    def forward(self, x, hypergraph):
        device = x.device
        hypergraph = hypergraph.to(device)

        if x.dim() == 2:
            x = x.view(1, -1, self.in_dim)

        attention_weights_all_layers = []  # 用于保存所有层的注意力权重

        for layer in self.layers:
            x = layer(x, hypergraph)
            if hasattr(layer, 'attention_weights') and layer.attention_weights is not None:
                # print(f"[HGCN] Layer attention weight shape: {layer.attention_weights.shape}")
                attention_weights_all_layers.append(layer.attention_weights.detach().clone())

        self.attention_weights = attention_weights_all_layers
        return x

    def get_attention_weights(self):
        return self.attention_weights

    def update_groups(self, new_num_groups):
        self.num_groups = new_num_groups
        for layer in self.layers:
            layer.update_groups(new_num_groups)

class HGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_groups):
        super(HGCNLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_groups = num_groups

        # 特征变换矩阵
        self.feature_transform = nn.Linear(in_dim, in_dim)

        # 使用超图卷积的概念进行组内特征聚合
        self.group_transform = nn.Linear(num_groups, in_dim)

        # 线性卷积层，用于最终的特征输出
        self.linear = nn.Linear(in_dim, out_dim)

        # print(f"HGCNLayer in_dim: {in_dim}")

        # 注意力机制，用于智能体特征的组内信息聚合
        self.attention = nn.MultiheadAttention(embed_dim=in_dim, num_heads=4, batch_first=True)

        # 注意力权重变量
        self.attention_weights = None

    def forward(self, x, hypergraph):
        device = x.device
        hypergraph = hypergraph.to(device)
        self.feature_transform = self.feature_transform.to(device)
        self.group_transform = self.group_transform.to(device)
        self.attention = self.attention.to(device)
        self.linear = self.linear.to(device)

        # x 的形状是 (batch_size, num_agents, feature_dim)
        # print(f"原始特征: {x}")
        batch_size, num_agents, _ = x.size()

        # 特征变换：线性变换 X -> X'
        x_transformed = F.relu(self.feature_transform(x))

        # 使用邻接矩阵进行卷积式聚合 (超图卷积)
        hypergraph_agg = torch.bmm(hypergraph, x_transformed)  # (batch_size, num_groups, feature_dim)

        # 使用注意力机制进行组内智能体间的信息共享
        # 每个智能体基于自己与组内其他成员的信息进行加权组合
        x_with_attention, attention_weights = self.attention(x, hypergraph_agg, hypergraph_agg)
        self.attention_weights = attention_weights.detach().clone()  # 保存注意力权重
        # print(f"[HGCNLayer] Attention weight shape: {attention_weights.shape}")

        # 对聚合后的特征通过线性层进行卷积变换
        x_combined = F.relu(self.linear(x_with_attention))
        return x_combined

    def update_groups(self, new_num_groups):
        if new_num_groups != self.num_groups:
            old_weight = self.group_transform.weight.data
            self.num_groups = new_num_groups
            new_group_transform = nn.Linear(new_num_groups, self.in_dim).to(old_weight.device)
            # 复制旧权重到新权重，如果新的组数更少，我们只复制需要的部分
            min_groups = min(old_weight.size(1), new_num_groups)
            new_group_transform.weight.data[:, :min_groups] = old_weight[:, :min_groups]
            self.group_transform = new_group_transform