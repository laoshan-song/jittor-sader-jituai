from pathlib import Path


def patch_file(path: Path, replacements: list[tuple[str, str]]) -> None:
    text = path.read_text()
    original = text
    for old, new in replacements:
        text = text.replace(old, new)
    if text != original:
        path.write_text(text)
        print(f"patched {path}")
    else:
        print(f"unchanged {path}")


def main() -> None:
    import jittor_geometric

    root = Path(jittor_geometric.__file__).resolve().parent

    patch_file(
        root / "ops" / "spmmcsr.py",
        [
            ("# Run the test\njt.flags.use_cuda=1\n", ""),
            (
                "    def execute(self,x,csr,trans_A,trans_B):\n"
                "        from jittor.compile_extern import cusparse_ops\n",
                "    def execute(self,x,csr,trans_A,trans_B):\n"
                "        if not jt.has_cuda:\n"
                "            raise RuntimeError("
                "\"SpmmCsr requires CUDA/cusparse. "
                "Use the CPU aggregate path instead.\")\n"
                "        jt.flags.use_cuda = 1\n"
                "        from jittor.compile_extern import cusparse_ops\n",
            ),
            (
                "    def grad(self, grad_output):\n"
                "        from jittor.compile_extern import cusparse_ops\n",
                "    def grad(self, grad_output):\n"
                "        if not jt.has_cuda:\n"
                "            raise RuntimeError(\"SpmmCsr backward requires CUDA/cusparse.\")\n"
                "        jt.flags.use_cuda = 1\n"
                "        from jittor.compile_extern import cusparse_ops\n",
            ),
        ],
    )

    patch_file(
        root / "ops" / "spmmcoo.py",
        [
            ("# Run the test\njt.flags.use_cuda=1\n", ""),
            (
                "    def execute(self,x,edge_index,edge_weight,trans_A,trans_B):\n"
                "        from jittor.compile_extern import cusparse_ops\n",
                "    def execute(self,x,edge_index,edge_weight,trans_A,trans_B):\n"
                "        if not jt.has_cuda:\n"
                "            raise RuntimeError("
                "\"SpmmCoo requires CUDA/cusparse. "
                "Use a CPU aggregation path instead.\")\n"
                "        jt.flags.use_cuda = 1\n"
                "        from jittor.compile_extern import cusparse_ops\n",
            ),
            (
                "    def grad(self, grad_output):\n"
                "        from jittor.compile_extern import cusparse_ops\n",
                "    def grad(self, grad_output):\n"
                "        if not jt.has_cuda:\n"
                "            raise RuntimeError(\"SpmmCoo backward requires CUDA/cusparse.\")\n"
                "        jt.flags.use_cuda = 1\n"
                "        from jittor.compile_extern import cusparse_ops\n",
            ),
        ],
    )

    patch_file(
        root / "ops" / "__init__.py",
        [
            (
                "from .getweight import getweight\n"
                "from .sampleprocessing import sampleprocessing\n"
                "from .gpuinitco import gpuinitco\n",
                "def getweight(*args, **kwargs):\n"
                "    from .getweight import getweight as _getweight\n"
                "    return _getweight(*args, **kwargs)\n\n"
                "def sampleprocessing(*args, **kwargs):\n"
                "    from .sampleprocessing import sampleprocessing as _sampleprocessing\n"
                "    return _sampleprocessing(*args, **kwargs)\n\n"
                "def gpuinitco(*args, **kwargs):\n"
                "    from .gpuinitco import gpuinitco as _gpuinitco\n"
                "    return _gpuinitco(*args, **kwargs)\n",
            ),
        ],
    )


if __name__ == "__main__":
    main()
