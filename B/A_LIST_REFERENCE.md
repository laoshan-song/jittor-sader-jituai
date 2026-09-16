# Accepted A-list reference

The independently supplied and accepted A-list archive is recorded as the
algorithm and delivery reference for this B-list package:

```text
archive: A榜参考contest1_sader_007.zip
SHA-256: d159f406a4b6376eb29ebe5b7d54c2481094706dd9f6c67987792cb62966e275
recorded score: 1.521072794155721
```

## Verifiable members

| Member | Bytes | SHA-256 |
| --- | ---: | --- |
| `README.md` | 8,424 | `844582a5eeb6ccd4de6e08c316e785d657fc12da3fe9d84488a328d5e770af8e` |
| `code/README.md` | 2,775 | `a205d7d7abe38c1798485515d1229480f3683b68d662a34cd91ec029fa49adff` |
| `base_result.zip` | 73,507,454 | `4c8fca6a041a94957b04a2df9f958d76098d06ab67093679fea766c364e3a28f` |
| `models/model_bpr32_prod_community_jittor.npz` | 14,989,914 | `449c45c3d32b477efa52410f5ec9024801942e264ba79873b981297005e3d256` |

The accepted package establishes the shared contract used here:

1. The reviewer supplies the official competition archive.
2. The package validates and loads its retained base and Jittor checkpoint.
3. Jittor scoring and fixed postprocessing generate the result archive.
4. The final archive is verified against the recorded SHA-256.
5. Raw training source and official-data execution commands are included for
   review and supplementary execution.

The B-list `verify` route follows the same retained-state contract. Its
`reproduce` route additionally executes the complete larger training graph,
fresh D3/D4 inference, and final MF32 training before the deterministic builder
constructs and verifies the recorded two-member result ZIP.
