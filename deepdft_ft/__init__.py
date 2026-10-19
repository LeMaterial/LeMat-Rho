"""DeepDFT (peterbjorgensen/DeepDFT) fine-tuning glue for LeMat-Rho.

Mirrors ``charge3net_ft/`` in structure: the data loader reuses ``charge3net_ft``'s
parquet helpers and adapts the per-sample shape to DeepDFT's dict contract.
"""
