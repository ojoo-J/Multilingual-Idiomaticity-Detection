# SemEval-2022-task2
Multilingual Idiomaticity Detection and Sentence Embedding 

## Data
Subtask A, zero-shot setting only

## MelBERT
Orig: https://github.com/jin530/MelBERT
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
### train
~~~
python main.py --model_type MELBERT --bert_model xlm-roberta-base
~~~
- cfg 파일에서 do_train = True, do_test = False, do_eval = True로 설정.
- data/dev.csv → data/test.csv로 이름 변경 후 사용.
### inference
~~~
python main.py --model_type MELBERT --bert_model {path of saves file}
~~~
- cfg 파일에서 do_train = False, do_test = True, do_eval = False로 설정.
- data/eval.csv → data/test.csv로 이름 변경 후 사용.
