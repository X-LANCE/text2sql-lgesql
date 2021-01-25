#coding=utf8
import sys, os, json, pickle, argparse, time
from argparse import Namespace
os.environ['NLTK_DATA'] = os.path.join(os.path.sep, 'root', 'nltk_data')
os.environ["STANZA_RESOURCES_DIR"] = os.path.join(os.path.sep, 'root', 'stanza_resources')
os.environ['EMBEDDINGS_ROOT'] = os.path.join(os.path.sep, 'root', '.embeddings')

import torch
from preprocess.process_dataset import process_tables, process_dataset
from preprocess.process_dataset_graph import process_dataset_graph
from preprocess.common_utils import Preprocessor
from utils.example import Example
from utils.batch import Batch
from model.model_utils import Registrable
from model.model_constructor import *

def preprocess_database_and_dataset(db_dir='database/', table_path='data/tables.json', dataset_path='data/dev.json'):
    tables = json.load(open(table_path, 'r'))
    dataset = json.load(open(dataset_path, 'r'))
    processor = Preprocessor(db_dir=db_dir, db_content=True)
    output_tables = process_tables(processor, tables)
    output_dataset = process_dataset(processor, dataset, output_tables)
    output_dataset = process_dataset_graph(processor, dataset, output_tables)
    return output_dataset, output_tables

def load_examples(dataset, tables):
    ex_list = []
    for ex in dataset:
        ex_list.append(Example(ex, tables[ex['db_id']]))
    return ex_list

parser = argparse.ArgumentParser()
parser.add_argument('--db_dir', default='database', help='path to db dir')
parser.add_argument('--table_path', default='data/tables.json', help='path to tables json file')
parser.add_argument('--dataset_path', default='data/dev.json', help='path to raw dataset json file')
parser.add_argument('--saved_model', default='saved_models/glove42B', help='path to saved model path')
parser.add_argument('--output_path', default='predicted_sql.txt', help='output predicted sql file')
parser.add_argument('--batch_size', default=20, type=int, help='batch size for evaluation')
parser.add_argument('--beam_size', default=5, type=int, help='beam search size')
args = parser.parse_args(sys.argv[1:])

dataset, tables = preprocess_database_and_dataset(db_dir=args.db_dir, table_path=args.table_path, dataset_path=args.dataset_path)
params = json.load(open(os.path.join(args.saved_model, 'params.json'), 'r'), object_hook=lambda d: Namespace(**d))
params.lazy_load = True # load PTM from AutoConfig instead of AutoModel.from_pretrained(...)
vocab_path = os.path.join(args.saved_model, 'vocab.txt')
Example.configuration(params.ptm, method=params.method, tables=tables)
dataset = load_examples(dataset, tables)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
model = Registrable.by_name('hetgnn-sql')(params, Example.trans).to(device)
check_point = torch.load(open(os.path.join(args.saved_model, 'model.bin'), 'rb'), map_location=device)
model.load_state_dict(check_point['model'])

start_time = time.time()
print('Start evaluating ...')
model.eval()
all_hyps = []
with torch.no_grad():
    for i in range(0, len(dataset), args.batch_size):
        current_batch = Batch.from_example_list(dataset[i: i + args.batch_size], device, train=False, method='hetgnn')
        hyps = model.parse(current_batch, args.beam_size)
        all_hyps.extend(hyps)
with open(args.output_path, 'w', encoding='utf8') as of:
    for idx, hyp in enumerate(all_hyps):
        best_ast = hyp[0].tree # by default, the top beam prediction
        pred_sql = Example.trans.ast_to_surface_code(best_ast, dataset[idx].db)
        of.write(pred_sql + '\n')
print('Evaluation costs %.4fs .' % (time.time() - start_time))
