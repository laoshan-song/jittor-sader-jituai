import sys

import jittor as jt
from jittor_geometric.nn import GCNConv
from jittor_geometric.nn.conv.gcn_conv import gcn_norm
from jittor_geometric.ops import cootocsc, cootocsr


def main() -> None:
    print("python", sys.executable)
    print("jittor", jt.__version__)
    print("has_cuda", jt.has_cuda)
    print("use_cuda", jt.flags.use_cuda)
    print("nvcc_path", jt.flags.nvcc_path)

    values = jt.array([1, 2, 3])
    print("sum", int(values.sum().item()))

    edge_index = jt.array([[0, 1, 2, 3], [1, 2, 3, 0]], dtype="int32")
    edge_index, edge_weight = gcn_norm(edge_index, num_nodes=4)
    csc = cootocsc(edge_index, edge_weight, 4)
    csr = cootocsr(edge_index, edge_weight, 4)

    x = jt.randn((4, 3))
    conv = GCNConv(3, 2, spmm=False)
    out = conv(x, csc, csr)
    print("gcn_output_shape", tuple(out.shape))
    print("environment ok")


if __name__ == "__main__":
    main()
