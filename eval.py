#coding=utf8
import sys, os, json, pickle, argparse, time
from argparse import Namespace
os.environ['NLTK_DATA'] = os.path.join(os.path.sep, 'root', 'nltk_data')
os.environ["STANZA_RESOURCES_DIR"] = os.path.join(os.path.sep, 'root', 'stanza_resources')
os.environ['EMBEDDINGS_ROOT'] = os.path.join(os.path.sep, 'root', '.embeddings')

import torch
from preprocess.dataset_process import process_tables, process_dataset
from utils.example import Example
from utils.batch import Batch
from model.model_utils import Registrable
from model.model_constructor import *

def preprocess_database_and_dataset(db_dir='database/', dataset_dir='data/'):
    tables = json.load(open(os.path.join(dataset_dir, 'tables.json'), 'r'))
    dataset = json.load(open(os.path.join(dataset_dir, 'dev.json'), 'r'))

    output_tables = process_tables(tables)
    output_dataset = process_dataset(dataset, output_tables, db_dir=db_dir)
    return output_dataset, output_tables

def load_examples(dataset, tables):
    ex_list = []
    for ex in dataset:
        ex_list.append(Example(ex, tables[ex['db_id']]))
    return ex_list

parser = argparse.ArgumentParser()
parser.add_argument('--db_dir', default='database', help='path to db dir')
parser.add_argument('--dataset_dir', default='data', help='path to dataset and tables')
parser.add_argument('--saved_model', default='saved_models/glove42B', help='path to saved model path')
parser.add_argument('--output_path', default='predicted_sql.txt', help='output predicted sql file')
parser.add_argument('--batch_size', default=20, type=int, help='batch size for evaluation')
parser.add_argument('--beam_size', default=5, type=int, help='beam search size')
args = parser.parse_args(sys.argv[1:])

dataset, tables = preprocess_database_and_dataset(db_dir=args.db_dir, dataset_dir=args.dataset_dir)
params = json.load(open(os.path.join(args.saved_model, 'params.json'), 'r'), object_hook=lambda d: Namespace(**d))
params.lazy_load = True # load PTM from AutoConfig instead of AutoModel.from_pretrained(...)
vocab_path = os.path.join(args.saved_model, 'vocab.txt')
Example.configuration(params.ptm, add_cls=False, test=True, tables=tables, processed=True,
    vocab_path=(vocab_path if os.path.exists(vocab_path) else None))
dataset = load_examples(dataset, tables)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
model = Registrable.by_name('ratsql_coarse2fine')(params, Example.trans).to(device)
check_point = torch.load(open(os.path.join(args.saved_model, 'model.bin'), 'rb'), map_location=device)
model.load_state_dict(check_point['model'])

start_time = time.time()
print('Start evaluating ...')
model.eval()
all_hyps = []
with torch.no_grad():
    for i in range(0, len(dataset), args.batch_size):
        current_batch = Batch.from_example_list(dataset[i: i + args.batch_size], device, train=False, method='ratsql_coarse2fine')
        hyps, select_mask = model.parse(current_batch, args.beam_size, mode='multitask')
        all_hyps.extend(hyps)
with open(args.output_path, 'w', encoding='utf8') as of:
    for idx, hyp in enumerate(all_hyps):
        best_ast = hyp[0].tree # by default, the top beam prediction
        pred_sql = Example.trans.ast_to_surface_code(best_ast, dataset[idx].db)
        of.write(pred_sql + '\n')
print('Evaluation costs %.4fs .' % (time.time() - start_time))
