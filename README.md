# CardioBench: Cardiovascular Diagnostic Benchmark

CardioBench is the first Level-4 (L4) clinical benchmark for cardiovascular specialties, designed to systematically quantify the diagnostic reasoning, evidence traceability, and safety (e.g., hallucination rate) of large language models (LLMs) in real-world clinical scenarios.

## Repository Structure
This repository is organized into two complementary parts:

### 1. CardioBench Evaluation Framework
A modular Python framework for running multi-tier evaluations (L1–L4) of LLMs on cardiovascular diagnostic tasks. It implements:
- **Multi-tier scoring** (Primary Diagnosis, Secondary Diagnosis, Differential Diagnosis)
- **Evidence traceability** (token-level F1 between generated and gold-standard evidence)
- **Hallucination detection** (flagging of unsupported clinical statements)
- **Subset-wise analysis** (performance breakdown across Typical, Mild, Critical, Rare cases)
- **Model-agnostic interface** (unified API wrapper for easy integration of new LLMs)

### 2. CardioDataset
A curated collection of **808 de-identified clinical vignettes** spanning 13 cardiovascular sub‑specialties, stratified into four clinical subsets:
- **Typical**: Inpatient cases with complete workups
- **Mild**: Outpatient cases with simple presentations
- **Critical**: High‑acuity, life‑threatening conditions
- **Rare**: Low‑prevalence cardiovascular diseases

## Key Features
- **Beyond accuracy**: Evaluates reasoning, evidence grounding, and safety—not just multiple‑choice correctness.
- **Real‑world stratification**: Mirrors clinical practice with cases of varying acuity, prevalence, and data richness.
- **Reproducible & extensible**: Open‑source code and standardized data format enable community benchmarking and iteration.

## Citation
If you use CardioBench in your research, please cite our paper (preprint link).

## License
- **Code**: MIT License
- **Data Access**: The benchmark dataset (CardioDataset) is available for non‑commercial academic research. Qualified researchers can request access by submitting a formal request to the corresponding author via [Email of corresponding author Yixiu Liang](mailto:tegkux@gmail.com). Data sharing is subject to a Data Use Agreement to ensure compliance with patient privacy regulations and institutional ethics approval (IRB No. B2024‑497(2), Zhongshan Hospital, Fudan University).
