diagnosis_list_prompt="""<Predicted Inferred Diagnosis List>
{}

<Standard Diagnostic Result>
{}

<Patient Medical History> (Analysis of the medical history is permitted if and only if **hyponym matching** is performed; before using the medical history, you must explicitly state that this condition is met)
{}

## Task Description
- Based on the <Standard Diagnostic Result>, apply the **core decision rules and steps** to determine whether there is a diagnosis name in the <Predicted Inferred Diagnosis List> that matches the <Standard Diagnostic Result>. If yes, identify one and return its position (index) in the list;
- **<Patient Medical History> may only be used under specific circumstances**: only when performing the determination of a **hyponym diagnosis** is it permitted to use and analyze the <Patient Medical History> to further confirm whether the predicted diagnosis is accurate; **in all other circumstances, reviewing or analyzing the <Patient Medical History> is prohibited**;
- **First principles**: throughout the determination process, everything must follow the **core decision rules and steps**, and it must be explicitly assumed that the provided standard diagnosis is absolutely correct; no other factors need to be considered;
- The diagnoses in the predicted list are independent of each other. Once a matching diagnosis is found during the determination process, there is no need to consider other diagnoses in the list (i.e., only one matching diagnosis needs to be found). At the same time, cases where multiple diagnoses in the list combined cover the standard diagnosis are not allowed.
- The standard diagnosis is absolutely correct, so medical history should be considered only when encountering a hyponym diagnosis. When the predicted diagnosis is a hypernym (broader term), it can be directly judged as inconsistent;

### **Core Decision Rules and Steps**
- Execute strictly according to the following code logic: iterate through the diagnosis list. Note that multiple elements in the list independently execute the following process, but once RETURN is encountered, all subsequent analysis must be terminated immediately and the answer must be given directly, ensuring rigor and non-interference.
- According to the code execution principle: Step 1 does not need to consider subsequent Steps 2/3/4; similarly, Step 2 does not need to consider Steps 3/4; and so on;
- **When comparing the current diagnosis, there is no need to process or consider other diagnosis elements in the Predicted Inferred Diagnosis List**
- Keep the analysis logically coherent, avoid using interrogative sentences as much as possible, avoid contradictions, and strictly follow the **first principles** requirements

```plaintext
FOR each (current diagnosis) IN Predicted Inferred Diagnosis List:
    # --- Step 1: Terminology normalization ---
    1.1 IF current diagnosis is directly consistent with the standard diagnosis in semantics or medical concept:
          RETURN a matching diagnosis exists (**immediately terminate all loops and thought processes**)  # emphasize that it must RETURN immediately here
    1.2 Normalize the standard diagnosis and the current diagnosis
        1.2.1 Terminology normalization conversion (e.g., AMI → acute myocardial infarction)
        1.2.2 Remove modifiers/pathological (disease course) staging descriptions: (including but not limited to: suspected, to be investigated, chronic, high-risk, moderate)
    1.3 **Update the normalized diagnoses back to the current diagnosis and the standard diagnosis respectively**
    1.4 IF current diagnosis == standard diagnosis: # Note: compare using the normalized diagnoses; no need to consider the impact of the original names (e.g., chronic pneumonia and pneumonia are also considered consistent after normalization and can be returned directly)
            RETURN a matching diagnosis exists (**immediately terminate all loops and thought processes**)  # emphasize that it must RETURN immediately here

    # --- Step 2: Hierarchical relationship determination ---
    2.1 IF current diagnosis is a hyponym of the standard diagnosis:
            IF medical history is allowed AND the medical history supports the hyponym diagnosis:
                RETURN a matching diagnosis exists (**immediately terminate all analysis and loops**) 
            ELSE:
                CONTINUE to the next diagnosis loop
    2.2 ELIF current diagnosis is a hypernym of the standard diagnosis:
            CONTINUE to the next diagnosis loop (forcibly exclude hypernym diagnoses)

    # --- Step 3: Conflict detection --- 
    3.1 IF there is an anatomic location/pathologic type/temporal logic conflict:
            CONTINUE to the next diagnosis loop

    # --- Step 4: Core element matching ---
    4.1 Split compound diagnoses: split them into independent core elements (e.g., "hypertensive nephropathy" → "hypertension" and "nephropathy");
    4.2 Extract the set S of core elements of the standard diagnosis
    4.3 Extract the set P of core elements of the current diagnosis
    4.4 IF P fully contains S:
            RETURN a matching diagnosis exists (**immediately terminate all analysis and loops**) 
        ELSE:
            CONTINUE to the next diagnosis loop

Note:
1. CONTINUE means abandon the current diagnosis and continue to the next diagnosis in the loop
2. RETURN means forcibly terminate the logic; all subsequent analysis and looping processes must be completely terminated; further analysis of other diagnoses is prohibited, **to avoid redundancy**;
3. All split core elements of the standard diagnosis must be covered to satisfy Step 4
```

### Response Format (You must strictly generate the response according to the following format; do not generate any other task-irrelevant content)
Whether there is a diagnosis in the <Predicted Inferred Diagnosis List> that matches the <Standard Diagnostic Result>: No/Yes
Diagnosis in the <Predicted Inferred Diagnosis List> that matches the <Standard Diagnostic Result> (confirmed to exclude all **hypernym diagnoses**): None/["xx"]
Position of the diagnosis judged as consistent in the <Predicted Inferred Diagnosis List> (0-based indexing): None/[0]"""



evidence_hallucination_prompt="""Please determine whether the <Diagnostic Information> is hallucinated information relative to the <Medical Record Information> based on the following two scenarios:
## Scenario (1): If the <Diagnostic Information> contains negative content such as "missing... examination," "no mention of... results," "no... symptoms," etc., and the negative content described in the <Diagnostic Information> exists in the <Medical Record Information>, then the <Diagnostic Information> is judged to be hallucinated information; otherwise, it is not hallucinated information. After completing the judgment, provide the result according to the reply format, and do not evaluate Scenario (2).
## Scenario (2): If the <Diagnostic Information> does not contain negative content such as "missing," directly check whether there is relevant description of the <Diagnostic Information> in the <Medical Record Information>. If no relevant description exists, then the <Diagnostic Information> is judged to be hallucinated information; otherwise, it is not hallucinated information. After completing the judgment, provide the result according to the reply format.
<Medical Record Information>
{}

<Diagnostic Information>
{}

【Reply Format (output the judgment result according to the format; no other content is needed)】:
Hallucination: Yes/No
"""


trace_query_prompt_o="""## Your task is to combine the content in <Question> and <Previous Information> (if provided), and determine the traceability result based on whether the knowledge points in the **user-specified segment** (<Segment to Be Traced>) clearly and explicitly reference the knowledge points in the <Evidence Set>;

## Answer requirements:
- If <Segment to Be Traced> is a general, summary statement, output the extraction result directly as: []
- If <Segment to Be Traced> is **advice or logical reasoning** based on <Previous Information> or the <Evidence Set>, output the extraction result directly as: []
- If <Segment to Be Traced> does **not explicitly** reference the knowledge points in the <Evidence Set>, output the extraction result directly as: []
- If multiple segments are referenced, all of them must be output;
- Return the final extraction result in the form of a Python list, for example: ['s1','s2']
- If the content does not include the knowledge points provided in the referenced segments, output the extraction result directly as: []
## User Input
### <Evidence Set>：
'''{}'''

### <Segment to Be Traced>：
'''{}'''
"""

#跑诊断结果prompt
raw_prompt="""{}
## Task Description
- Given the patient information, please complete the clinical diagnostic process according to **clinical medical reasoning** (including primary and secondary diagnoses, and differential diagnosis).
- You need to focus on analyzing all of the patient’s negative and positive features, including all physical examination and ancillary test results, summarize the corresponding diagnostic evidence, then dynamically integrate the chain of evidence, and provide specific diagnostic conclusions through clinical reasoning.

## **Requirements for Diagnostic Evidence**:
  - **Completeness**: Ensure that all evidence relevant to the current diagnosis is listed, and **bold** the evidence with **high specificity**, marking it **separately**;
  - **Consistency**: Use the original wording as much as possible for all evidence descriptions; avoid modification, deletion, or summarization; fabrication of factual evidence is prohibited under any circumstances;
  - **Order**: Strictly classify and output evidence by **History, Physical Examination, Ancillary Tests**, and indicate the specific category; organize in order to make the results clearer;
    - Example: ["History: xxxx", "Physical Examination: xxxxxxx"];
    - Only the above three categories are allowed; do not use other fields;
  - **Evidence for Differential Diagnosis**: For differential diagnoses, list supporting and non-supporting evidence, summarized into the corresponding fields as follows:
    - **Supporting evidence**: Classify evidence by **History, Physical Examination, Ancillary Tests**; for related evidence, provide a combined description (e.g., signs + imaging features);
    - **Non-supporting evidence**:
      - Classify as “absolute exclusion evidence” and “evidence pending verification”; “evidence pending verification” must be labeled, example: ["Evidence pending verification (History): xxxx"];
      - Must consider and explain the limitations of negative results (e.g., insufficient test sensitivity);
    - Key tests that have not been completed must be labeled as “tests not completed”;
    - If there is currently no non-supporting evidence, output an empty list ('[]') is allowed;

## Clinical Medical Reasoning
### Primary Diagnosis
- Directly points to the root cause of the patient’s current clinical manifestations; use standard ICD-10 disease names;
- Etiologic subtype: Based on the primary diagnosis, further refine the subtype characteristics of the disease;

### Secondary Diagnoses
- Generally refer to one or more diseases in the patient’s past history, with relatively less harm to the patient’s health, and that do not occupy a major position during treatment;

### Differential Diagnosis
- Based on the preliminary diagnosis, consider other diseases with symptoms and signs similar to the patient’s, and rule out these possibilities through further examinations and assessments;
- Requires divergent and reverse thinking, able to infer possible related diseases from the patient’s clinical manifestations;
- A good differential diagnosis helps improve diagnostic accuracy and **reduce misdiagnosis and missed diagnosis**;

### Basis for Selecting Differential Diagnoses (According to Clinical Reasoning Methods)
  - Differentiate from diseases that share similarities with this disease and are in adjacent anatomical locations;
  - Consider common and frequently occurring diseases;
  - Consider diseases that may lead to serious consequences: diseases that may cause severe outcomes if not diagnosed and treated promptly, to ensure life-threatening conditions are not missed;
  - When necessary, consider regional or seasonal diseases (e.g., influenza);
  - If some secondary diagnoses may affect the differential diagnosis: determine their possible impact on comorbid symptoms (e.g., infection), disease manifestations and progression, and the differential process;
  - **If there is already clear non-supporting evidence**, the current diagnosis to be differentiated should be directly excluded to avoid an ineffective differential diagnosis process and output;

## Response Format
- When generating the final answer: you must output the diagnostic results directly in Json format (example below), and do not output any other irrelevant content;
- {"Preliminary diagnosis": {"Primary diagnosis": {"Name": "XXX", "Etiologic subtype": "XXX", "Diagnostic evidence": ["Evidence 1", "Evidence 2"]}, "Secondary diagnoses": [{"Name": "XXX", "Diagnostic evidence": ["Evidence 1", "Evidence 2"]}]}, "Differential diagnoses": [{"Name": "XXX", "Selection rationale": "Consider rare but potentially fatal diseases", "Supporting evidence": ["Evidence 1", "Evidence 2"], "Non-supporting evidence": [], "Excludable": true, "Recommended corresponding tests": []}, {"Name": "YYY", "Selection rationale": "Consider differential diagnosis of adjacent organ lesions", "Supporting evidence": ["Evidence 1", "Evidence 2"], "Non-supporting evidence": ["Evidence 1"], "Excludable": false, "Recommended corresponding tests": ["Recommendation 1", "Recommendation 2"]}]}
"""