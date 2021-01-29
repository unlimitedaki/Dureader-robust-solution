# Dureader-robust-solution

## 文件目录

``` shell
-external-liberaries 
-model
-dataset 
squad.py
squad_metrics.py
evaluate.py
model.py
processor.py
run_dureader_robust_roberta.sh
run_dureader_robust_roberta_large.py
```

## 环境

```
torch-1.4.0+cu92
transformers-2.7.0
```

## 参数

```shell
--data_dir 数据集目录
--train_file 训练文件在数据集目录下的相对位置
--dev_file
--test_file
--output_dir 模型和log输出目录
--save_model_name 每个实验的模型在输出目录下的位置
--origin_model 首次训练加载的模型在输出目录下位置
--target_model 只在测试和仅验证时中有用,读取对应的model,可选 current_model & best_model
# store_true 参数，后面不需要参数值，有就是True，没有就是False
--do_train
--do_test 
--do_eval
--do_finetune
# 其他超参数:
--max_seq_length
--max_answer_length
--gradient_accumulation_steps
--num_train_epochs
--adam_epsilon
--learning_rate
--warmup_steps
--train_batch_size
--eval_batch_size
--n_best_size 
--threads 数据预处理的线程数
```

### 训练

```shell
--do_train 训练train_file
--do_eval会在每一个epoch输出验证结果，验证dev_file
--do_finetune 会取current_model结果继续finetune，将num_train_epochs减去现有的epoch
```

### 验证

```shell
--do_eval 测试dev_file，输出分数和结果文件
--target_model 验证的模型，可选best_model&current_model
```

### 测试

```shell
--do_test 测试test_file 只输出结果文件
--target_model 验证的模型，可选best_model&current_model
```

### 多轮finetune

```shell
--do_train 
--origin_model 已经预训练的模型路径，从这个模型开始进一步finetune,默认是中文reoberta—large
```





