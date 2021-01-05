#coding=utf8
import sys, os, time, json, gc
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from argparse import Namespace
from utils.args import init_args
from utils.hyperparams import hyperparam_path
from utils.initialization import *
from utils.example import Example
from utils.batch import Batch
from utils.optimization import set_optimizer
from model.model_utils import Registrable
from model.model_constructor import *

# initialization params, output path, logger, random seed and torch.device
args = init_args(sys.argv[1:], task='ratsql_coarse2fine')
exp_path = hyperparam_path(args, task='ratsql_coarse2fine')
if not os.path.exists(exp_path):
    os.makedirs(exp_path)
logger = set_logger(exp_path, args.testing)
set_random_seed(args.seed)
device = set_torch_device(args.device)
logger.info("Initialization finished ...")
logger.info("Output path is %s" % (exp_path))
logger.info("Random seed is set to %d" % (args.seed))
logger.info("Use GPU with index %s" % (args.device) if args.device >= 0 else "Use CPU as target torch device")

# load dataset and vocabulary
start_time = time.time()
if args.read_model_path:
    params = json.load(open(os.path.join(args.read_model_path, 'params.json')), object_hook=lambda d: Namespace(**d))
    args.ptm = params.ptm
Example.configuration(args.ptm, processed=args.preprocess, add_cls=True) # set up the grammar, transition system, evaluator and tables
train_dataset, dev_dataset = Example.load_dataset('train'), Example.load_dataset('dev')
logger.info("Load dataset and database finished, cost %.4fs ..." % (time.time() - start_time))
logger.info("Dataset size: train -> %d ; dev -> %d" % (len(train_dataset), len(dev_dataset)))
sql_trans, evaluator = Example.trans, Example.evaluator
args.word_vocab, args.relation_num = len(Example.word_vocab), len(Example.relative_position_vocab)

# model init, set optimizer
if args.read_model_path:
    model = Registrable.by_name('ratsql_coarse2fine')(params, sql_trans).to(device)
    check_point = torch.load(open(os.path.join(args.read_model_path, 'model.bin'), 'rb'), map_location=device)
    model.load_state_dict(check_point['model'])
    logger.info("Load saved model from path: %s" % (args.read_model_path))
else:
    json.dump(vars(args), open(os.path.join(exp_path, 'params.json'), 'w'), indent=4)
    model = Registrable.by_name('ratsql_coarse2fine')(args, sql_trans).to(device)
    if args.ptm is None:
        ratio = Example.word2vec.load_embeddings(model.encoder.input_layer.word_embed, Example.word_vocab, device=device)
        logger.info("Init model and word embedding layer with a coverage %.2f" % (ratio))
# logger.info(str(model))
num_training_steps = ((len(train_dataset) + args.batch_size - 1) // args.batch_size) * args.max_epoch
num_warmup_steps = int(num_training_steps * args.warmup_ratio)
logger.info('Total training steps: %d;\t Warmup steps: %d' % (num_training_steps, num_warmup_steps))
optimizer, scheduler = set_optimizer(model, args, num_warmup_steps, num_training_steps)
if args.read_model_path and args.load_optimizer:
    optimizer.load_state_dict(check_point['optim'])

def decode(choice, output_path, suffix, acc_type='sql', add_fscore=True):
    assert acc_type in ['beam', 'ast', 'sql'] and choice in ['train', 'dev']
    model.eval()
    dataset = train_dataset if choice == 'train' else dev_dataset
    all_hyps, all_select = [], []
    with torch.no_grad():
        for i in range(0, len(dataset), args.batch_size):
            current_batch = Batch.from_example_list(dataset[i: i + args.batch_size], device, train=False, method='ratsql_coarse2fine')
            raw_table_reverse_mappings, raw_column_reverse_mappings = current_batch.table_reverse_mappings, current_batch.column_reverse_mappings
            hyps, select_mask = model.parse(current_batch, args.beam_size, mode='multitask')
            all_hyps.extend(hyps)
            if add_fscore:
                all_select.append([select_mask.int().tolist(), current_batch.select_mask.int().tolist(),
                    raw_table_reverse_mappings, raw_column_reverse_mappings,
                    current_batch.table_lens.tolist(), current_batch.column_lens.tolist()])
        acc_output_path = os.path.join(output_path, choice + '_acc.' + suffix)
        acc = evaluator.acc(all_hyps, dataset, acc_output_path, acc_type=acc_type, etype='match', choice=choice)
        if add_fscore:
            fscore_output_path = os.path.join(output_path, choice + '_fscore.' + suffix)
            fscore = evaluator.fscore(all_select, dataset, fscore_output_path, only_error=True, return_metric='acc')
    torch.cuda.empty_cache()
    gc.collect()
    if add_fscore:
        return acc, fscore
    else:
        return acc

if not args.testing:
    nsamples, best_result = len(train_dataset), {'dev_acc': 0.}
    train_index, step_size = np.arange(nsamples), args.batch_size // args.grad_accumulate
    aux_train_index = np.arange(nsamples)
    logger.info('Start training ......')
    for i in range(args.max_epoch):
        start_time = time.time()
        epoch_loss, epoch_prune_loss, count = 0, 0, 0
        np.random.shuffle(train_index)
        np.random.shuffle(aux_train_index)
        model.train()
        for j in range(0, nsamples, step_size):
            count += 1

            cur_dataset = [train_dataset[k] for k in aux_train_index[j: j + step_size]]
            current_batch = Batch.from_example_list(cur_dataset, device, train=True, method='graph_pruning', ls=args.label_smoothing, loss_function=args.loss_function)
            prune_loss = model(current_batch, mode='graph_pruning')
            prune_loss *= args.prune_coeffi
            epoch_prune_loss += prune_loss.item()
            prune_loss.backward()

            cur_dataset = [train_dataset[k] for k in train_index[j: j + step_size]]
            current_batch = Batch.from_example_list(cur_dataset, device, train=True, method='ratsql_coarse2fine', ls=args.label_smoothing, loss_function=args.loss_function, min_rate=args.min_rate, max_rate=args.max_rate)
            loss = model(current_batch, mode='text2sql') # see utils/batch.py for batch elements
            epoch_loss += loss.item()
            loss.backward()

            # print("Minibatch loss: %.4f/%.4f" % (loss.item(), prune_loss.item()))
            if count == args.grad_accumulate or j + step_size >= nsamples:
                count = 0
                model.pad_embedding_grad_zero()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        logger.info('Training: \tEpoch: %d\tTime: %.4f\tTraining loss: %.4f/%.4f' % (i, time.time() - start_time, epoch_loss, epoch_prune_loss))
        torch.cuda.empty_cache()
        gc.collect()

        if i <= args.eval_after_epoch: # avoid unnecessary evaluation
            continue

        start_time = time.time()
        dev_acc, fscore = decode('dev', exp_path, 'iter' + str(i), acc_type='sql')
        logger.info('Evaluation: \tEpoch: %d\tTime: %.4f\tDev acc: %.4f' % (i, time.time() - start_time, dev_acc))
        logger.info('Evaluation: \tEpoch: %d\tTime: %.4f\tDev pruning acc/recall acc: %.4f/%.4f' % (i, time.time() - start_time, fscore[0], fscore[1]))
        if dev_acc > best_result['dev_acc']:
            best_result['dev_acc'], best_result['iter'], best_result['fscore'] = dev_acc, i, fscore
            torch.save({
                'epoch': i, 'model': model.state_dict(),
                'optim': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, open(os.path.join(exp_path, 'model.bin'), 'wb'))
            logger.info('NEW BEST MODEL: \tEpoch: %d\tDev acc: %.4f\tPruning acc/recall acc: %.4f/%.4f' % (i, dev_acc, fscore[0], fscore[1]))

    check_point = torch.load(open(os.path.join(exp_path, 'model.bin'), 'rb'))
    model.load_state_dict(check_point['model'])
    # train_acc, fscore = decode('train', exp_path, 'iter' + str(best_result['iter']), acc_type='ast')
    # logger.info('FINAL BEST RESULT: \tEpoch: %d\tPruning acc/recall acc: %.4f/%.4f' % (best_result['iter'], fscore[0], fscore[1]))
    dev_acc_beam = decode('dev', exp_path, 'iter' + str(best_result['iter']) + '.beam' + str(args.beam_size), acc_type='beam', add_fscore=False)
    logger.info('FINAL BEST RESULT: \tEpoch: %d\tDev acc/Beam acc: %.4f/%.4f' % (best_result['iter'], best_result['dev_acc'], dev_acc_beam))
    logger.info('FINAL BEST RESULT: \tEpoch: %d\tDev pruning acc/recall acc: %.4f/%.4f' % (best_result['iter'], best_result['fscore'][0], best_result['fscore'][1]))
else:
    logger.info('Start evaluating ...')
    start_time = time.time()
    # train_acc, train_fscore = decode('train', args.read_model_path, 'eval', acc_type='ast')
    # logger.info("Evaluation costs %.2fs ; Train dataset pruning acc/recall acc is %.4f/%.4f ." % (time.time() - start_time, train_fscore[0], train_fscore[1]))
    dev_acc, dev_fscore = decode('dev', args.read_model_path, 'eval', acc_type='sql')
    dev_acc_beam = decode('dev', args.read_model_path, 'eval.beam' + str(args.beam_size), acc_type='beam', add_fscore=False)
    logger.info("Evaluation costs %.2fs ; Dev dataset exact match acc/inner beam acc is %.4f/%.4f ." % (time.time() - start_time, dev_acc, dev_acc_beam))
    logger.info("Evaluation costs %.2fs ; Dev dataset pruning acc/recall acc is %.4f/%.4f ." % (time.time() - start_time, dev_fscore[0], dev_fscore[1]))
