# jittor-sader-jituai

第六届计图人工智能挑战赛项目仓库。当前已完成热身赛一：基于 Cora 引文网络的 GCN 节点分类。

## 当前结果

- 任务：热身赛一，Cora 节点分类
- 框架：Jittor + JittorGeometric
- 本地最佳验证集准确率：0.8120
- 平台提交状态：已通过
- 提交包：`submissions/warmup1-result.zip`

## 目录结构

```text
.
├── docs/
│   └── Jittor-competition-summary.md
├── scripts/
│   └── patch_jittor_geometric_cpu.py
├── submissions/
│   └── warmup1-result.zip
├── warmups/
│   └── cora_gcn/
│       ├── data/cora.pkl
│       ├── gcn.py
│       ├── README.md
│       └── result.json
├── ENVIRONMENT.md
├── env.sh
├── requirements.txt
└── verify_env.py
```

## 快速复现

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install git+https://github.com/AlgRUC/JittorGeometric.git
python scripts/patch_jittor_geometric_cpu.py
source env.sh
python verify_env.py
cd warmups/cora_gcn
python gcn.py
```

运行完成后会生成 `warmups/cora_gcn/result.json`。提交包格式见 `submissions/warmup1-result.zip`。

## 说明

当前开发机 GPU/CUDA 不可用，所以 `env.sh` 默认走 CPU 路线，并禁用了 MPI 与 Jittor 自动 CUDA 下载。正式赛训练建议迁移到可用 GPU 机器后再打开 CUDA。
