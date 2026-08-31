"""Warm-up 1: Cora node classification with a two-layer GCN."""

from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path

import jittor as jt
from jittor import nn
import numpy as np
from jittor_geometric.nn import GCNConv
from jittor_geometric.nn.conv.gcn_conv import gcn_norm
from jittor_geometric.ops import cootocsc, cootocsr


DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "data" / "cora.pkl"
DEFAULT_OUTPUT_PATH = Path("result.json")


class GraphData:
    """Container for graph tensors and sparse adjacency formats."""


class GCNNet(nn.Module):
    """Two-layer GCN for semi-supervised node classification."""

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_dim: int,
        dropout: float,
        use_spmm: bool,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(num_features, hidden_dim, spmm=use_spmm)
        self.conv2 = GCNConv(hidden_dim, num_classes, spmm=use_spmm)

    def execute(self, graph: GraphData) -> jt.Var:
        """Return logits with shape [num_nodes, num_classes]."""
        x = nn.relu(self.conv1(graph.x, graph.csc, graph.csr))
        x = nn.dropout(x, self.dropout, is_train=self.training)
        return self.conv2(x, graph.csc, graph.csr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a GCN on Cora and write warm-up predictions."
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.8)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU execution even when Jittor reports CUDA is available.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and Jittor random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    jt.misc.set_global_seed(seed)


def load_raw_data(data_path: Path) -> dict:
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}\n"
            "Download or copy the competition release file to "
            "warmup/cora_gcn/data/cora.pkl, or pass --data-path."
        )

    with data_path.open("rb") as file:
        return pickle.load(file)


def build_graph(raw_data: dict) -> GraphData:
    """Convert raw Cora arrays into Jittor tensors and sparse graph formats."""
    graph = GraphData()
    graph.x = jt.array(raw_data["x"].astype(np.float32))
    graph.y = jt.array(raw_data["y"].astype(np.int64))
    graph.edge_index = jt.array(raw_data["edge_index"].astype(np.int64))
    graph.train_mask = jt.array(raw_data["train_mask"])
    graph.val_mask = jt.array(raw_data["val_mask"])
    graph.test_mask = jt.array(raw_data["test_mask"])

    row_sum = jt.clamp(graph.x.sum(dim=1, keepdims=True), min_v=1e-12)
    graph.x = graph.x / row_sum

    num_nodes = graph.x.shape[0]
    edge_index, edge_weight = gcn_norm(
        graph.edge_index,
        None,
        num_nodes,
        improved=False,
        add_self_loops=True,
    )

    with jt.no_grad():
        graph.csc = cootocsc(edge_index, edge_weight, num_nodes)
        graph.csr = cootocsr(edge_index, edge_weight, num_nodes)

    return graph


def train_one_epoch(model: GCNNet, graph: GraphData, optimizer: nn.Optimizer) -> float:
    model.train()
    logits = model(graph)[graph.train_mask]
    labels = graph.y[graph.train_mask]
    loss = nn.cross_entropy_loss(logits, labels)
    optimizer.step(loss)
    return float(loss.item())


def evaluate(model: GCNNet, graph: GraphData) -> tuple[float, float]:
    model.eval()
    logits = model(graph)
    scores = []

    for mask in [graph.train_mask, graph.val_mask]:
        pred, _ = jt.argmax(logits[mask], dim=1)
        labels = graph.y[mask]
        scores.append(float((pred == labels).float32().mean().item()))

    return scores[0], scores[1]


def save_predictions(model: GCNNet, graph: GraphData, raw_data: dict, output: Path) -> int:
    model.eval()
    logits = model(graph)
    pred, _ = jt.argmax(logits, dim=1)
    test_indices = np.where(raw_data["test_mask"])[0]

    result = {str(int(index)): int(pred[int(index)].item()) for index in test_indices}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    return len(result)


def main() -> None:
    args = parse_args()
    jt.flags.use_cuda = 0 if args.cpu or not jt.has_cuda else 1
    set_seed(args.seed)

    raw_data = load_raw_data(args.data_path)
    graph = build_graph(raw_data)
    model = GCNNet(
        num_features=int(raw_data["num_features"]),
        num_classes=int(raw_data["num_classes"]),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        use_spmm=bool(jt.flags.use_cuda),
    )
    optimizer = nn.Adam(
        params=model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, graph, optimizer)
        train_acc, val_acc = evaluate(model, graph)
        best_val_acc = max(best_val_acc, val_acc)

        if epoch % args.log_interval == 0 or epoch == args.epochs:
            print(
                "Epoch: "
                f"{epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Best Val Acc: {best_val_acc:.4f}"
            )

    print(f"\nFinal Val Acc: {best_val_acc:.4f}")
    count = save_predictions(model, graph, raw_data, args.output)
    print(f"Predictions saved to {args.output}")
    print(f"Predicted {count} test nodes")


if __name__ == "__main__":
    main()
