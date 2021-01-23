#coding=utf8
import os, json, pickle, argparse, sys, time
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from preprocess.common_utils import Preprocessor

def process_dataset_graph(processor, dataset, tables, output_path=None):
    processed_dataset = []
    for idx, entry in enumerate(dataset):
        entry['graph'] = processor.prepare_graph(entry, tables[entry['db_id']])
        processed_dataset.append(entry)
    print('In total, process %d samples .' % (len(processed_dataset)))
    if output_path is not None:
        # serialize preprocessed dataset
        pickle.dump(processed_dataset, open(output_path, 'wb'))
    return processed_dataset

if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--dataset_path', type=str, required=True, help='dataset path')
    arg_parser.add_argument('--table_path', type=str, required=True, help='processed table path')
    arg_parser.add_argument('--output_path', type=str, required=True, help='output preprocessed dataset')
    args = arg_parser.parse_args()

    processor = Preprocessor(db_dir='data/database', db_content=True)
    # loading database and dataset
    tables = pickle.load(open(args.table_path, 'rb'))
    dataset = pickle.load(open(args.dataset_path, 'rb'))
    start_time = time.time()
    dataset = process_dataset_graph(processor, dataset, tables, args.output_path)
    print('Dataset preprocessing costs %.4fs .' % (time.time() - start_time))
