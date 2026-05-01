# NeuroVeil: Multi-Modal AI for Long COVID Detection

Brain MRI Radiomics + Blood Transcriptomics combined AI system for Long COVID (PASC) classification.

## Project Structure

```
neuroveil/
├── data/
│   ├── mri/
│   │   └── Dryad_final.xlsx          # 173 hasta, 682 beyin bolgesi (FreeSurfer)
│   └── blood/
│       ├── raw/
│       │   ├── GSE226260_combined.csv.gz      # Kohort 1 gen ifadesi (228 ornek)
│       │   ├── GSE226260_additional.csv.gz    # Kohort 2 gen ifadesi (103 ornek)
│       │   └── GSE226260_family.soft.gz       # Metadata (hasta etiketleri)
│       └── processed/
│           ├── blood_X.csv            # Islenmis gen ifadesi matrisi (189x23864)
│           └── blood_y.csv            # Hedef degisken (PASC=1, Control=0)
│
├── pipelines/
│   ├── mri_pipeline.py               # Modul 1: MRI Radyomiks (SVM, %74.29)
│   └── blood_pipeline.py             # Modul 2: Kan Gen Ifadesi (SVM, %97.37)
│
├── experiments/
│   ├── optuna_search.py              # Optuna hiperparametre optimizasyonu
│   ├── god_mode_optimizer.py         # RFECV brute-force sutun eleme
│   ├── correlation_table.py          # Korelasyon esik degeri karsilastirmasi
│   ├── test_models.py                # Alternatif model karsilastirmasi
│   ├── test_optuna_ensemble.py       # Stacking ensemble deneyi
│   └── check_soft.py                 # GEO SOFT metadata parser
│
├── utils/
│   └── prepare_blood_data.py         # GSE226260 veri hazirlama scripti
│
├── results/
│   ├── neuroveil_results.png         # MRI modulu grafikleri
│   └── blood_module_results.png      # Kan modulu grafikleri
│
├── .venv/                            # Python sanal ortam
└── README.md                         # Bu dosya
```

## Modules

| Module | Data Source | Features | Model | Accuracy |
|--------|-----------|----------|-------|----------|
| MRI Radiomics | Dryad (173 hasta) | 682 brain regions | LASSO + SVM | **74.29%** |
| Blood Transcriptomics | GSE226260 (189 hasta) | 23,864 genes | LASSO + SVM | **97.37%** |
| Ensemble (TODO) | Both | Combined | Meta-Learner | TBD |

## Data Sources
- **MRI:** Dryad Repository - Post-COVID Condition brain MRI study
- **Blood:** NCBI GEO GSE226260 - "Long COVID involves activation of proinflammatory and immune exhaustion pathways" (Barouch Lab, 2026)
