# SemEval-2022-task2
Multilingual Idiomaticity Detection and Sentence Embedding 

## MelBERT
Orig: https://github.com/jin530/MelBERT

### Data
Subtask A, zero-shot setting only

### requirements
boto3==1.16.63 </br>
nltk==3.5 </br>
numpy==1.20.0 </br>
requests==2.25.1 </br>
scikit-learn==0.24.1 </br>
scipy==1.6.0 </br>
torch==1.6.0 </br>
torchvision==0.7.0 </br>
tqdm==4.56.0 </br>
transformers==4.2.2 </br>

### Model Type
- **MELBERT**: SPV + MIP / only target sentence / 
- **MELBERT_CONTEXT**: SPV + MIP / context o / 
- **MELBERT_SPV**: SPV / only target sentence / 
- **MELBERT_SPV_CONTEXT**: SPV / context o / 


### Run
~~~
opython main.py --model_type {model type name} --bert_model xlm-roberta-base
~~~




