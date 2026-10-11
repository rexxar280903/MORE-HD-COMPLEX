# MADBase data audit

- Dataset: `MADBASE-AUC-V1`
- Official source: https://datacenter.aucegypt.edu/shazeem/
- Snapshot SHA-256: `8262dca2d3db6205e0112586dd9a02c35b0b371da292aab0a15dc92976793f78`
- Inventory: 60,000 official-train and 10,000 official-test images, 28×28, ten balanced classes.
- Writers: 600 train and 100 test; overlap = 0.
- Raw polarity: white background/dark foreground; canonical loader applies `255 - raw`.
- Blank images: 0 train, 0 test.
- Exact duplicate groups within a split: 39.
- Exact duplicate groups crossing train/test: 12.
- Exact duplicate groups with conflicting labels: 0.
- Cross-split dHash candidates scanned at Hamming distance ≤1: 532516.
- Near-exact cross-split flags after requiring pixel MAE ≤2: 3879; these are review flags, not exclusions.
- License: official download is public, but no standard license text was found on the official page; confirm research/redistribution terms before confirmatory publication.

No model output or final-test performance was inspected during this audit.
