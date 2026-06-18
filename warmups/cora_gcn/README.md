# 赛道一热身赛 - 示例代码

本赛题提供示例代码框架，提供数据加载、模型定义、训练步骤等功能。

选手可以基于示例代码填充注释为 TODO 的部分完成该赛题：


## 使用说明

1. 配置环境：参考 [JittorGeometric 官方安装指南](https://github.com/AlgRUC/JittorGeometric?tab=readme-ov-file#installation)，安装 Jittor、JittorGeometric 和依赖。
2. 数据集文件 `cora.pkl` 应放置于 `data/` 目录下
3. 完成 `gcn.py` 中的 TODO
4. 运行 `python gcn.py` 进行训练和预测

## 数据集说明

数据集文件 `data/cora.pkl` 为 pickle 格式，包含以下字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `x` | numpy array (2708, 1433) | 节点特征矩阵 |
| `y` | numpy array (2708,) | 节点标签（测试集标签为 -1） |
| `edge_index` | numpy array (2, num_edges) | 边列表 |
| `train_mask` | numpy bool array (2708,) | 训练集掩码 |
| `val_mask` | numpy bool array (2708,) | 验证集掩码 |
| `test_mask` | numpy bool array (2708,) | 测试集掩码 |
| `num_classes` | int | 类别数（7） |
| `num_features` | int | 特征维度（1433） |
