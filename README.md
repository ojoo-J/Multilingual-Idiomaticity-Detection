# Effective Idiomaticity Detection with Consideration at Different Levels of Contextualization
**NAACL SemEval 2022** [[📝 Paper](https://aclanthology.org/2022.semeval-1.17/)] [[👩🏻‍💻 Project Page (kor)](https://lavish-cent-5cf.notion.site/Effective-Idiomaticity-Detection-with-Consideration-at-Different-Levels-of-Contextualization-16b43df64a2780a5a8c8f57a8057219f?pvs=4)]

## Summary
This research was presents a solution to [SemEval-2022 Task 2: Multilingual Idiomaticity Detection and Sentence Embedding](https://aclanthology.org/2022.semeval-1.13.pdf). The task focuses on determining whether a Multi-Word Expression (MWE) within a sentence is used idiomatically or not in a multilingual setting. Identifying idiomatic expressions in sentences using Large Language Models (LLMs) remains a challenging problem. This study explores how to effectively perform sentence embedding to address this task.

## Core Idea
<div align="center">
    <img src="https://github.com/user-attachments/assets/f689283c-3a5d-4995-b6d5-18584a75d8dd" alt="image" width="500">
</div>

The expression “wet blanket,” when interpreted through the literal meanings of its individual words, refers to “a soaked cover.” However, in the context of a sentence, it is often used idiomatically to mean “a person who spoils the mood.” In other words, if the meaning derived from the sentence context differs from the meaning based solely on the combination of the individual words, it can be identified as an idiomatic expression. Building on this idea, when a Multi-Word Expression (MWE) is given, wouldn’t it be possible to effectively capture idiomaticity by generating semantic embeddings that combine the contextual embeddings of each word with their static embeddings (representing the literal combination of word meanings)?

## Methods
<div align="center">
    <img src="https://github.com/user-attachments/assets/10f87f8a-39ea-4671-a345-670e40fdc8bc" alt="image" width="600">
</div>

We propose a framework for embedding MWEs and their related sentences that utilizes both contextualized and static representations to maximize semantic information. For more details, please refer to the paper.

## Results
<div align="center">
    <img src="https://github.com/user-attachments/assets/e6ba8522-a425-4fe9-a678-2a71e69079d0" alt="image" width="500">
</div>

## run
You can download the data from [here](https://sites.google.com/view/semeval2022task2-idiomaticity#h.d8q8fzxwq2cl).
```
python main.py
```
