# Medical Dataset Description Document 
## 1. Document Overview 
This document provides a detailed description of the structure and field meanings of the medical dataset. This dataset includes the original patient medical records, the gold standard diagnosis results provided by doctors, and the input questions for the model. 


## 2. Overview of Data Structures 
Each piece of data is a complete JSON object, and it includes the following core components 

| Field Name | Description                                                  |
| ---------- | ------------------------------------------------------------ |
| seq_id     | Data unique identifier, used for data traceability           |
| raw_query  | The complete original medical record of the patient, including the chief complaint, current medical history, past medical history, personal history, family history, physical examination, and auxiliary examinations, all providing the complete clinical information. |
| ans_label  | The authoritative diagnosis result provided by the doctor based on the patient's medical records serves as the gold standard for evaluating the performance of the model. |
| question   | The model is provided with input questions, which consist of the patient's original medical records and standardized diagnostic task prompts. It is clearly stipulated that the model should complete the diagnosis following the clinical reasoning process. |