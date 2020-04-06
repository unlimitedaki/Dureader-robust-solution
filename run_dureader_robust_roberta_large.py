import argparse
import os
import sys
import time
import re
if os.path.exists("external-libraries"):
    sys.path.append('external-libraries')
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from transformers import BertTokenizer,BertModel
from transformers import AdamW,get_linear_schedule_with_warmup
from transformers import BertPreTrainedModel
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.nn import CrossEntropyLoss
from tqdm import trange, tqdm

import json
import logging
import os
from functools import partial
from multiprocessing import Pool, cpu_count

from tqdm import tqdm

from processor import load_and_cache_examples,SquadResult,SquadFeatures,SquadExample
from model import BertForQuestionAnswering
import evaluate as Eval
from squad_metrics import compute_predictions_logits,squad_evaluate
import timeit

# from ...file_utils import is_tf_available, is_torch_available
from transformers.tokenization_bert import whitespace_tokenize
# from .utils import DataProcessor


# if is_torch_available():
import torch
from torch.utils.data import TensorDataset

# if is_tf_available():
#     import tensorflow as tf

logger = logging.getLogger(__name__)

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def evaluate(args, model, tokenizer, device,data_type,prefix=""):
    # if data_type == 'test':
    #     dataset,examples,features = load_and_cache_examples(args, tokenizer, data_type, output_examples=True,prefix = prefix)
    # else:
    dataset, examples, features = load_and_cache_examples(args, tokenizer, data_type = data_type, output_examples=True,prefix = prefix)

    output_dir = os.path.join(args.output_dir,args.save_model_name)
    if not os.path.exists(output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(output_dir)

    # args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    all_results = []
    start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }


            example_indices = batch[3]

            outputs = model(**inputs)
            # pdb.set_trace()
        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)

            output = [to_list(output[i]) for output in outputs]

            # Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
            # models only use two.
            if len(output) >= 5:
                start_logits = output[0]
                start_top_index = output[1]
                end_logits = output[2]
                end_top_index = output[3]
                cls_logits = output[4]

                result = SquadResult(
                    unique_id,
                    start_logits,
                    end_logits,
                    start_top_index=start_top_index,
                    end_top_index=end_top_index,
                    cls_logits=cls_logits,
                )

            else:
                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)

            all_results.append(result)

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

    # Compute predictions
    output_prediction_file = os.path.join(output_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(output_dir, "nbest_predictions_{}.json".format(prefix))


    output_null_log_odds_file = None

    predictions = compute_predictions_logits(
        examples,
        features,
        all_results,
        args.n_best_size,
        args.max_answer_length,
        args.do_lower_case,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        args.verbose_logging,
        False,
        args.null_score_diff_threshold,
        tokenizer,
    )

    # Compute the F1 and exact scores.
    if data_type == 'test':
        return 
    dev_dir = os.path.join(args.data_dir,args.dev_file)
    dev = json.load(open(dev_dir,'r'))
    prediction = json.load(open(output_prediction_file)) 
    F1, EM, TOTAL, SKIP = Eval.evaluate(dev,prediction)
    
    return F1, EM


def train(args):
    output_dir = os.path.join(args.output_dir,args.save_model_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logfilename = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+" "+args.save_model_name+".log.txt"
    fh = logging.FileHandler(os.path.join(output_dir,logfilename), mode='a', encoding='utf-8')
    fh.setLevel(logging.INFO)
    # ch = logging.StreamHandler(sys.stdout)
    # ch.setLevel(logging.INFO)
    logger.addHandler(fh)
    # logger.addHandler(ch)

    model_dir = os.path.join("model",'chinese_roberta_wwm_large_ext_pytorch')
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    train_dataset= load_and_cache_examples(args, tokenizer, data_type = 'train',output_examples=False,prefix =args.train_file.split('.')[0] )
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    
    # setup device
    if args.use_tpu :# Colab TPU is not better than GPU
        import torch_xla
        import torch_xla.core.xla_model as xm
        import torch_xla.debug.metrics as met
        import torch_xla.distributed.parallel_loader as pl
        import torch_xla.distributed.xla_multiprocessing as xmp
        import torch_xla.utils.utils as xu
        device = xm.xla_device()
    else:
        device = torch.device('cuda:0')
    
    # model
    if args.do_finetune:
        status_dir = os.path.join(output_dir,"status.json")
        status = json.load(open(status_dir,'r'))
        current_model = os.path.join(output_dir, "current_model")
        model = BertForQuestionAnswering.from_pretrained(current_model)
        
    else:
        origin_dir = os.path.join(args.output_dir,args.origin_model)
        model = BertForQuestionAnswering.from_pretrained(origin_dir)
        status = {}
        status['best_epoch'] = 0
        status['best_EM'] = 0.0
        status['best_F1'] = 0.0
        status['current_epoch']  = 0
        # status['global_step'] = 0
        
    model.to(device)
    
    # Prepare optimizer and schedule (linear warmup and decay)
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    tr_loss = 0.0
    # global_step = 0
    model.zero_grad()
    epochs_trained = 0
    train_iterator = trange(epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    # F1,EM = evaluate(args,model,tokenizer,device)
    # logger.info("Dev F1 = %s, EM = %s on epoch %s",str(F1),str(EM),str(-1))
    # Train!
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        tr_loss = 0
        
        
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(device) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
            }

            outputs = model(**inputs)
            # model outputs are always tuple in transformers (see doc)
            loss = outputs[0]
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step() 
                model.zero_grad()
            if (step + 1)% args.check_loss_step == 0 or step == len(train_dataloader):
                avg_loss = tr_loss/(step+1)
                logger.info("\t average_step_loss=%s @ step = %s on epoch = %s",str(avg_loss),str(step+1),str(epoch+1))
                
        # F1 , EM  = 11,22
        if args.do_eval:
            F1,EM = evaluate(args,model,tokenizer,device,data_type = 'dev',prefix = args.dev_file.split('.')[0])
            logger.info("Dev F1 = %s, EM = %s on epoch %s",str(F1),str(EM),str(epoch+1))
            # save the best model 
            output_dir = os.path.join(args.output_dir,args.save_model_name)
            if F1 > status['best_F1']:
                status['best_F1'] = F1
                status['best_EM'] = EM
                status['best_epoch'] = epoch
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                best_model_dir = os.path.join(output_dir,"best_model")
                # output_dir = os.path.join(output_dir, 'checkpoint-{}'.format(epoch + 1))
                if not os.path.exists(best_model_dir):
                    os.makedirs(best_model_dir)
                model_to_save.save_pretrained(best_model_dir)
                logger.info("best epoch %d has been saved to %s",epoch,best_model_dir)
            # save current model
        status['current_epoch'] = epoch
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_dir = os.path.join(args.output_dir,args.save_model_name)
        current_model_dir = os.path.join(output_dir,"current_model")
        
        if not os.path.exists(current_model_dir):
            os.makedirs(current_model_dir)
        model_to_save.save_pretrained(current_model_dir)
        logger.info("epoch %d has been saved to %s",epoch,current_model_dir)
        # save status
        status_dir = os.path.join(output_dir,"status.json")
        json.dump(status,open(status_dir,'w',encoding = 'utf8'))
        
        # p len(example.context_text)+len(example.question_text)+len(example.answer_text)

def test(arges):
    output_dir = os.path.join(args.output_dir,args.save_model_name)
    #device
    if args.use_tpu :
        import torch_xla
        import torch_xla.core.xla_model as xm
        import torch_xla.debug.metrics as met
        import torch_xla.distributed.parallel_loader as pl
        import torch_xla.distributed.xla_multiprocessing as xmp
        import torch_xla.utils.utils as xu
        device = xm.xla_device()
    else:
        device = torch.device('cuda:0')
    
    model_dir = os.path.join("model",'chinese_roberta_wwm_large_ext_pytorch')
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    status_dir = os.path.join(output_dir,"status.json")
    status = json.load(open(status_dir,'r'))
    current_model = os.path.join(output_dir, args.target_model)
    model = BertForQuestionAnswering.from_pretrained(current_model)
    model.to(device)
    if args.do_eval:
        evaluate(args,model,tokenizer,device,data_type = 'dev',prefix = args.dev_file.split('.')[0])
    else:
        evaluate(args,model,tokenizer,device,data_type = 'test',prefix = args.test_file.split('.')[0])


import pdb

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
# data arguments
parser.add_argument("--data_dir",type = str,default = "data/sampledata/demo")
parser.add_argument("--train_file",type= str,default = "demo_train.json")
parser.add_argument("--dev_file",type= str,default = "demo_dev.json")
parser.add_argument("--test_file",type = str)
parser.add_argument("--output_dir",type = str,default = 'model')
parser.add_argument("--save_model_name",type = str,default = "")
parser.add_argument("--origin_model",type = str,default = "chinese_roberta_wwm_large_ext_pytorch", help = "origin model dir for training")
# hyper parameters
parser.add_argument("--model_name_or_path",type = str,default = 'chinese-roberta')
parser.add_argument("--local_rank",type =int,default = -1)
parser.add_argument("--max_seq_length",type = int ,default = 384,help = "max sequence length of examples")
parser.add_argument("--max_answer_length",default=30,type=int)
parser.add_argument("--doc_stride",default=128,type=int,help="When splitting up a long document into chunks, how much stride to take between chunks.",)
parser.add_argument("--max_query_length",default=64,type=int,help="The maximum number of tokens for the question. Questions longer than this will be truncated to this length.",)
parser.add_argument("--gradient_accumulation_steps",type=int,default=4,help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--num_train_epochs",default=5,type=int)
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
parser.add_argument("--train_batch_size", default=6, type=int, help="Batch size for training.")
parser.add_argument("--eval_batch_size", default=6, type=int, help="Batch size for training.")
parser.add_argument("--n_best_size",default=20,type=int,help="The total number of n-best predictions to generate in the nbest_predictions.json output file.")
parser.add_argument("--null_score_diff_threshold",type=float,default=0.0,help="If null_score - best_non_null is greater than the threshold predict null.")
parser.add_argument("--check_loss_step",default = 800,type = int,help = "output current average loss of training")
# settings
parser.add_argument("--do_train",action="store_true",default = False,help = "Whether to train")
parser.add_argument("--do_finetune",action = "store_true", default = False)
parser.add_argument("--do_eval",action="store_true",default = False , help= "Whether to evaluate")
parser.add_argument("--do_test",action = "store_true",default = False,help = "Whether to test")
parser.add_argument("--threads", type=int, default=1, help="multiple threads for converting example to features")
parser.add_argument("--overwrite_cache",type = bool,default = False)
parser.add_argument("--use_tpu",type = bool,default = False)
parser.add_argument("--n_gpu",type=int , default = 1)
parser.add_argument("--verbose_logging",action="store_true",help="If true, all of the warnings related to data processing will be printed. A number of warnings are expected for a normal SQuAD evaluation.",)
parser.add_argument("--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.")
parser.add_argument("--target_model",type = str)
# parser.add_arg
args = parser.parse_args()

if args.do_train:
    train(args)
if args.do_test:
    test(args)
if not args.do_train and args.do_eval:
    test(args)
#loading data

