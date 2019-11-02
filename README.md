# hugginface_albert
hugginface albert model and its tokenizer <br>
The albert repo transfer weights from tf-hub module and can run `examples/run_squad_albert.py` to reproduce author's performance. <br>
support for albert_base, albert_large, albert_xlarge <br>

First Clone this repo. <br>
Second, prepare your training data {Training data path} and testing data{Testing data path}.  <br>
Current status: you can just run the squad task on albert-base model by using the below code on examples/ <br>

Run on Squad 1.1 task (albert base) <br>
EM 79.89 , F1 score 87.98 

The paper report the performance is below: <br>
EW 82.3 F1 score 89.3

```python run_squad_albert.py --train_file {Training data path} --predict_file {Testing data path} --model_type albert --output_dir ./example_do_lowercase_lamb --do_train --do_eval  --overwrite_output_dir --do_lower_case --evaluate_during_training --warmup_steps 384  --save_steps 1000 --logging_steps 1000 --num_train_epochs 3```

if you want albert_large or albert_xlarge, you need to download the pytorch_state_dict on here https://drive.google.com/drive/folders/1OZuoBw0cNkPLFXdoz4xTZrTXdRHO7hUI?usp=sharing <br> and put on ```examples/pytorch_model_state_dict``` and set new a argument below as ```--model_dict_pretrain ./pytorch_model_state_dict/{model_file name}```. also adjust the config file set to ```--config_pretrain ./config/{alber_xxx.json}``` then you can use it for big model.


