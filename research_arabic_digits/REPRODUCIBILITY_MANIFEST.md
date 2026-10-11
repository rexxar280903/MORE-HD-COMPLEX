# Manifest Reproduksi — Studi Digit Arab (MADBase)

Dibuat otomatis oleh `research_arabic_digits/tools/finalize_g6.py` pada 2026-10-11T06:52:57+07:00. Status G6: **CLOSED**.

## Kode

- Commit yang direkam semua run: `188bab6a068afcb02d76b18bd23cf5449943ed2d`
- HEAD saat manifest dibuat: `188bab6a068afcb02d76b18bd23cf5449943ed2d`
- Diff engine `core/` terhadap commit run: NONE
- File belum di-commit/termodifikasi saat manifest dibuat: lihat `REPRODUCIBILITY_MANIFEST.json` (`code.untracked_or_modified_now`) dan `DEVIATION_LOG.md` DEV-02.

## Ringkasan run

- Run utama COMPLETED: 450/450 (attempt: {1: 450})
- Jalur B: {'COMPLETED': 90}; fit baseline klasik: {'COMPLETED': 240} (+120 baris chance analitik)
- Budget: clustering 900, supervised 940 (FROZEN)

## Artefak dan SHA-256

### dataset_manifest

| Path | Bytes | SHA-256 |
|---|---:|---|
| `research_arabic_digits/research_data/manifests/raw_files.sha256` | 191 | `99f474e8af096e0efa04ccf982851eee6b4aa3b413fb6e8b6952463cad6d5b69` |
| `research_arabic_digits/research_data/manifests/dataset_inventory.json` | 2289 | `9fee2ffa8c915e0a82fbdeffe2992ab4fad7cd85dfcdd4ed1f9d0574036bc195` |
| `research_arabic_digits/research_data/manifests/label_mapping.json` | 529 | `9d1e5c93e75322977a9e10f61612767475ca8524dbdf801fb7b94f786dfe21f9` |
| `research_arabic_digits/research_data/manifests/duplicate_audit.csv` | 92474 | `669739f23d691b027f13a9231c7509c335a73e49dc85919aed77f1229827fc20` |
| `research_arabic_digits/research_data/manifests/data_audit_report.md` | 1059 | `b0ca8eb8de9dacefdd5fba1a28c61fce456256f59b28185cc6976d503b626f25` |
| `research_arabic_digits/DATASET_CARD.md` | 10337 | `a7b2f182adaef4c21b0a69e989d739521c5b541ab4fa533073d920cb31d848da` |

### split_manifests

| Path | Bytes | SHA-256 |
|---|---:|---|
| `research_arabic_digits/splits/confirmatory_seed101.json` | 699343 | `7d84804418c27ca33bace8a8f59dd340a7759e086dbc71f72f75564585f04433` |
| `research_arabic_digits/splits/confirmatory_seed202.json` | 699645 | `09aad74d30eb07ab0f56c67c98cbd23a28b6312168024dc72bfddb425b7693c9` |
| `research_arabic_digits/splits/confirmatory_seed303.json` | 699343 | `24eaa97681f597b098decf88f9acc56a4adcb1f196851f45374eae98eed50049` |
| `research_arabic_digits/splits/confirmatory_seed404.json` | 699544 | `8c318a74909c77347e2311a64fce9fab588ac5a3b45fb86d30142b5f2bd5e1f0` |
| `research_arabic_digits/splits/confirmatory_seed505.json` | 699342 | `06771c886213f6b98856158ff7d7a07c5da1e7e37101b0c1e78b57c4eafeb600` |
| `research_arabic_digits/splits/pilot_seed42.json` | 702601 | `9eccafb608759e319977a9cb3c69c39da4d447e8e4b8870a22457341e038caaf` |
| `research_arabic_digits/splits/split_registry.json` | 2731 | `b6ad3ef7cc95e801afc1aeb3fc07ed7bdadf8443bc806a5da432c532abf27e90` |

### protocol

| Path | Bytes | SHA-256 |
|---|---:|---|
| `research_arabic_digits/PROTOCOL_FREEZE.json` | 2157 | `5f1ead3f5bd25c59fa4402b2f05c4e2d0982f3aa853c5b08144cdb5383f307a0` |
| `research_arabic_digits/RESEARCH_PROTOCOL.md` | 6677 | `5d703b18c0eafe69d8ea22fa8889e72ed403fe5ca3d2615925a618f77d4aa85c` |
| `research_arabic_digits/STATISTICAL_ANALYSIS_PLAN.md` | 6691 | `5359778ea3a8ad1cbada8365276d9e246e758e59c311dd2da53b182fb8137d05` |
| `research_arabic_digits/DEVIATION_LOG.md` | 4882 | `4d8641fc213f0de5e90861efe304873b1fd35ad0a15499ae7c0c04a6c2e2684f` |

### results

| Path | Bytes | SHA-256 |
|---|---:|---|
| `research_arabic_digits/RESULTS.md` | 31442 | `73285ef66d75dad24010a41c8d01c678b67226403013330817c1dbe3f23822bf` |

### run_plan

| Path | Bytes | SHA-256 |
|---|---:|---|
| `research_arabic_digits/research_data/reports/confirmatory_run_plan.json` | 243455 | `d6808a70c167b71d527ca7e423d4a6999e24c4abb817b9babef789dfa6d1273d` |
| `research_arabic_digits/research_data/reports/queue8_plan.json` | 291 | `6c396c1740ae1bf495f93b9c555dfc128b7c13017df03c623a732cf571b671b6` |

### budget_decision

| Path | Bytes | SHA-256 |
|---|---:|---|
| `research_arabic_digits/research_data/pilot/convergence_budget_decision.json` | 4370 | `2cf051192907e376cbf520501998dd3c4e4e8c4862ed2f0a51816bec9582396d` |
| `research_arabic_digits/research_data/pilot/convergence_check.json` | 12564 | `4f701380349b59208551b0004020623bab26a8cb8a31ae98339e01a12966d3f7` |

### admission_report

| Path | Bytes | SHA-256 |
|---|---:|---|
| `research_arabic_digits/research_data/reports/final_admission.json` | 192 | `b1db529ae6d9843a9bb6a6df9c5a48dae7ed8c691e338bbc07315bec389aa82b` |
| `research_arabic_digits/research_data/reports/full_readiness_audit.json` | 1480 | `842587053505cd556cf7046f7824803d4fa83c63d857fc9cd5f21584d1ba1495` |

### consolidated_workbooks

| Path | Bytes | SHA-256 |
|---|---:|---|
| `research_arabic_digits/research_data/consolidated/ablation_90runs.xlsx` | 35623570 | `75bf161c7a4089b5f0509beee0d70279757674e3478bccddabbb87e3a87f7dda` |
| `research_arabic_digits/research_data/consolidated/more_reference_120runs.xlsx` | 48856354 | `24244408b05c6623d4f2df49560bb4eea744a2074c6d708cd1b14deec2241f9e` |
| `research_arabic_digits/research_data/consolidated/primary_240runs.xlsx` | 95753309 | `f5e0beed439331ef1ec97b3bfa0551f48a7908480f5e051c0cd7141d652e81b0` |

### analysis_tables

| Path | Bytes | SHA-256 |
|---|---:|---|
| `research_arabic_digits/research_data/analysis/MORE_HD_analysis_tables.xlsx` | 153897 | `232ed828232a37f072b3a45543a267fe9e07cdfb9c82289573d423e68dbb4bfa` |
| `research_arabic_digits/research_data/analysis/ablation_contrasts.csv` | 35695 | `1a7b47451104a38c2bbcd0dd89f6966ebdd1015ac2b49338ae0ab1e8393734be` |
| `research_arabic_digits/research_data/analysis/baseline_comparison.csv` | 39141 | `f7be4eac4ad82bcf243e3c5b952e5a6096d5ed1af28a0067a817f4fea3169a8a` |
| `research_arabic_digits/research_data/analysis/g1_11_six_nine.csv` | 6810 | `496e7753db0737d956bc22d7364cbe4569d38ca9ba2354fbed838650d0757bb4` |
| `research_arabic_digits/research_data/analysis/mcnemar_primary.csv` | 7361 | `d088e2c9012eb45fc2d77a505c38a209f98e5c5a624f46308c03c0ec70a7403d` |
| `research_arabic_digits/research_data/analysis/more_comparison.csv` | 50569 | `eb79f817c06bf44194c3879b1d54cd05a5f39f8bdc82f922a45301eff3a1be79` |
| `research_arabic_digits/research_data/analysis/paired_primary_D_minus_A.csv` | 37149 | `7c72047dd07a466411c78adbf2b1067d27c4f8bc6dbbeaaf39564a818368c9ea` |
| `research_arabic_digits/research_data/analysis/seed_aggregation.csv` | 50769 | `a0d09fea5f4946a85d92a73eaecfefdeaff94e8aebf4bc82971088443157ab07` |

### environment

| Path | Bytes | SHA-256 |
|---|---:|---|
| `research_arabic_digits/requirements-lock-py313.txt` | 340 | `587047f603050dfb1bcad4a8e8d63cbf28380ba677e039a750db177ca55e9c4e` |
| `research_arabic_digits/research_data/confirmatory_machine/environment_pretraining.json` | 2031 | `65c0cc5a5d660c9c9bef000c4591f4be5e6062872c55deda61c85e05c8df02a1` |
| `research_arabic_digits/research_data/confirmatory_machine/hardware_pretraining.json` | 260 | `458e0a47036970840859f47ee807b54e111eefefbee148399c609a8331d191ce` |

### paper_numbers

| Path | Bytes | SHA-256 |
|---|---:|---|
| `research_arabic_digits/paper/build/numbers.json` | 19666 | `d691c74fabdbe4e2b12b19a4e50372888d99ef48c8e22c1168af4afd0acfae69` |
| `research_arabic_digits/paper/build/tables.json` | 24614 | `8ed4a618968d6abc4322e0e4ccdb58d67cdb0e776954ff1f3d072d40bdf95e0a` |
| `research_arabic_digits/paper/build/runs_all.csv` | 93624 | `3d34361769019e42cd346f3066b8a1c9374d51c4ac4f4cd7c0539d1b0f6784bb` |

Data mentah MADBase tidak termasuk paket ini (kebijakan `LOCAL_RESEARCH_ONLY_NO_RAW_REDISTRIBUTION`); verifikasi unduhan resmi dengan `research_data/manifests/raw_files.sha256`.
