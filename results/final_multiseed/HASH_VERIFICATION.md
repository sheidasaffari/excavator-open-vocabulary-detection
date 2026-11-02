# Multi-seed hash verification

**Result: PASS**

Acceptance required three unique model-tensor hashes and three unique prediction hashes for every dataset within each family.

| Family | Model tensors | MOCS predictions | CIS predictions | Positive-control predictions | Confuser predictions |
|---|---:|---:|---:|---:|---:|
| Curated | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| Random | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |

Full SHA-256 values and artifact paths are stored in `HASH_VERIFICATION.json`.
