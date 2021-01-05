#coding=utf8
import sys, tempfile, os
import numpy as np
from asdl.sql.sql_transition_system import SelectColumnAction, SelectTableAction
from evaluator.evaluation import evaluate, build_foreign_key_map_from_json, build_valid_col_units, rebuild_sql_val, rebuild_sql_col, eval_exec_match
from evaluator.evaluation import Evaluator as Engine
from process_sql import get_schema, Schema, get_sql

class Evaluator():

    def __init__(self, transition_system, tables, train_gold_file='data/train_gold.sql', 
            dev_gold_file='data/dev_gold.sql', database_dir='data/database'):
        super(Evaluator, self).__init__()
        self.transition_system = transition_system
        self.tables = tables
        self.train_gold_file = train_gold_file
        self.dev_gold_file = dev_gold_file
        with open(self.train_gold_file, 'r') as f:
            self.train_dbs = [l.strip().split('\t')[1].strip() for l in f.readlines() if len(l.strip()) > 0]
        with open(self.dev_gold_file, 'r') as f:
            self.dev_dbs = [l.strip().split('\t')[1].strip() for l in f.readlines() if len(l.strip()) > 0]
        self.database_dir = database_dir
        self.kmaps = build_foreign_key_map_from_json('data/tables.json')
        self.engine = Engine()
        self.acc_dict = {
            "sql": self.sql_acc, # use golden sql file in data/ directory
            "ast": self.ast_acc, # some samples are skipped in the original dataset, cannot use the full version of golden.sql
            "beam": self.beam_acc, # if the correct answer exist in beam, assume the result is true
        }

    def acc(self, pred_hyps, dataset=[], output_path=None, acc_type='sql', etype='match', choice='dev'):
        assert acc_type in self.acc_dict and choice in ['train', 'dev'] and etype in ['match', 'exec']
        acc_method = self.acc_dict[acc_type]
        return acc_method(pred_hyps, dataset, output_path, etype, choice)

    def beam_acc(self, pred_hyps, dataset, output_path, etype, choice):
        assert len(pred_hyps) == len(dataset), 'Number of predictions %d is not equal to the number of references %d' % (len(pred_hyps), len(dataset))
        scores = {}
        for each in ['easy', 'medium', 'hard', 'extra', 'all']:
            scores[each] = [0, 0.] # first is count, second is total score
        results = []
        for idx, pred in enumerate(pred_hyps):
            question = dataset[idx].ex['question']
            t = dataset[idx].db
            gold_sql = self.transition_system.ast_to_surface_code(dataset[idx].ast, t)
            for b_id, hyp in enumerate(pred):
                pred_sql = self.transition_system.ast_to_surface_code(hyp.tree, t)
                score, hardness = self.single_acc(pred_sql, gold_sql, t['db_id'], etype)
                if int(score) == 1:
                    scores[hardness][0] += 1
                    scores[hardness][1] += 1.0
                    scores['all'][0] += 1
                    scores['all'][1] += 1.0
                    results.append((hardness, question, gold_sql, pred_sql, b_id, True))
                    break
            else:
                scores[hardness][0] += 1
                scores['all'][0] += 1
                pred_sql = self.transition_system.ast_to_surface_code(pred[0].tree, t)
                results.append((hardness, question, gold_sql, pred_sql, 0, False))
        for each in scores:
            accuracy = scores[each][1] / float(scores[each][0]) if scores[each][0] != 0 else 0.
            scores[each].append(accuracy)
        with open(output_path, 'w', encoding='utf8') as of:
            for item in results:
                of.write('Level: %s\n' % (item[0]))
                of.write('Question: %s\n' % (item[1]))
                of.write('Golden SQL: %s\n' %(item[2]))
                of.write('Pred SQL (%s): %s\n' % (item[4], item[3]))
                of.write('Equal: %s\n\n' % (item[5]))
            for each in scores:
                of.write('Level %s: %s\n' % (each, scores[each]))
        return scores['all'][2]

    def single_acc(self, pred_sql, gold_sql, db, etype):
        """
            @return:
                score(float): 0 or 1, etype score
                hardness(str): one of 'easy', 'medium', 'hard', 'extra'
        """
        db_name = db
        db = os.path.join(self.database_dir, db, db + ".sqlite")
        schema = Schema(get_schema(db))
        g_sql = get_sql(schema, gold_sql)
        hardness = self.engine.eval_hardness(g_sql)
        try:
            p_sql = get_sql(schema, pred_sql)
        except:
            # If p_sql is not valid, then we will use an empty sql to evaluate with the correct sql
            p_sql = {
            "except": None,
            "from": {
                "conds": [],
                "table_units": []
            },
            "groupBy": [],
            "having": [],
            "intersect": None,
            "limit": None,
            "orderBy": [],
            "select": [
                False,
                []
            ],
            "union": None,
            "where": []
            }
        kmap = self.kmaps[db_name]
        g_valid_col_units = build_valid_col_units(g_sql['from']['table_units'], schema)
        g_sql = rebuild_sql_val(g_sql)
        g_sql = rebuild_sql_col(g_valid_col_units, g_sql, kmap) # kmap: map __tab.col__ to pivot __tab.col__
        p_valid_col_units = build_valid_col_units(p_sql['from']['table_units'], schema)
        p_sql = rebuild_sql_val(p_sql)
        p_sql = rebuild_sql_col(p_valid_col_units, p_sql, kmap)
        if etype == 'exec':
            score = float(eval_exec_match(db, pred_sql, gold_sql, p_sql, g_sql))
        if etype == 'match':
            score = float(self.engine.eval_exact_match(p_sql, g_sql))
        return score, hardness

    def compare_sql_based_on_schema(self, pred_hyps, dataset, acc_flag, recall_acc_flag):
        pred_asts = [hyp[0].tree for hyp in pred_hyps]
        ref_asts = [ex.ast for ex in dataset]
        dbs = [ex.db for ex in dataset]
        sql_acc = []
        for pred, ref, db in zip(pred_asts, ref_asts, dbs):
            if pred is not None:
                pred_sql = self.transition_system.ast_to_surface_code(pred, db)
                ref_sql = self.transition_system.ast_to_surface_code(ref, db)
                score, _ = self.single_acc(pred_sql, ref_sql, db['db_id'], 'match')
                sql_acc.append(int(score))
            else:
                sql_acc.append(0)
        total_num = float(len(dataset))
        schema_true_sql_true = sum(map(lambda x, y: 1 if x == 1 and y == 1 else 0, acc_flag, sql_acc)) / total_num
        schema_true_sql_false = sum(map(lambda x, y: 1 if x == 1 and y == 0 else 0, acc_flag, sql_acc)) / total_num
        schema_false_sql_true = sum(map(lambda x, y: 1 if x == 0 and y == 1 else 0, acc_flag, sql_acc)) / total_num
        schema_false_sql_false = sum(map(lambda x, y: 1 if x == 0 and y == 0 else 0, acc_flag, sql_acc)) / total_num

        recall_schema_true_sql_true = sum(map(lambda x, y: 1 if x == 1 and y == 1 else 0, recall_acc_flag, sql_acc)) / total_num
        recall_schema_true_sql_false = sum(map(lambda x, y: 1 if x == 1 and y == 0 else 0, recall_acc_flag, sql_acc)) / total_num
        recall_schema_false_sql_true = sum(map(lambda x, y: 1 if x == 0 and y == 1 else 0, recall_acc_flag, sql_acc)) / total_num
        recall_schema_false_sql_false = sum(map(lambda x, y: 1 if x == 0 and y == 0 else 0, recall_acc_flag, sql_acc)) / total_num
        return (schema_true_sql_true, schema_true_sql_false, schema_false_sql_true, schema_false_sql_false), \
            (recall_schema_true_sql_true, recall_schema_true_sql_false, recall_schema_false_sql_true, recall_schema_false_sql_false)

    def compare_schema_in_ast(self, pred_hyps, dataset, output_path=None):
        """ Compare the used schema items in predicted and golden sql ast """
        pred_asts = [hyp[0].tree for hyp in pred_hyps]
        ref_asts = [ex.ast for ex in dataset]
        dbs = [ex.db for ex in dataset]
        schema_acc, sql_acc = [], []
        for pred, ref, db in zip(pred_asts, ref_asts, dbs):
            if pred is not None:
                pred_tables, pred_columns = set(), set()
                pred_acts = self.transition_system.get_actions(pred)
                for act in pred_acts:
                    if isinstance(act, SelectColumnAction):
                        pred_columns.add(act.column_id)
                        if act.column_id < len(db['column_names']):
                            pred_tables.add(db['column_names'][act.column_id][0])
                    elif isinstance(act, SelectTableAction):
                        pred_tables.add(act.table_id)
                    else: pass
                ref_tables, ref_columns = set(), set()
                ref_acts = self.transition_system.get_actions(ref)
                for act in ref_acts:
                    if isinstance(act, SelectColumnAction):
                        ref_columns.add(act.column_id)
                        if act.column_id < len(db['column_names']):
                            ref_tables.add(db['column_names'][act.column_id][0])
                        else:
                            print('Use column not in available tables')
                    elif isinstance(act, SelectTableAction):
                        ref_tables.add(act.table_id)
                    else: pass
                if ref_tables == pred_tables and ref_columns == pred_columns:
                    schema_acc.append(1)
                else:
                    schema_acc.append(0)
                pred_sql = self.transition_system.ast_to_surface_code(pred, db)
                ref_sql = self.transition_system.ast_to_surface_code(ref, db)
                score, _ = self.single_acc(pred_sql, ref_sql, db['db_id'], 'match')
                sql_acc.append(int(score))
            else:
                schema_acc.append(0)
                sql_acc.append(0)
        total_num = float(len(dataset))
        schema_true_sql_true = sum(map(lambda x, y: 1 if x == 1 and y == 1 else 0, schema_acc, sql_acc)) / total_num
        schema_true_sql_false = sum(map(lambda x, y: 1 if x == 1 and y == 0 else 0, schema_acc, sql_acc)) / total_num
        schema_false_sql_true = sum(map(lambda x, y: 1 if x == 0 and y == 1 else 0, schema_acc, sql_acc)) / total_num
        schema_false_sql_false = sum(map(lambda x, y: 1 if x == 0 and y == 0 else 0, schema_acc, sql_acc)) / total_num
        return schema_true_sql_true, schema_true_sql_false, schema_false_sql_true, schema_false_sql_false

    def ast_acc(self, pred_hyps, dataset, output_path, etype, choice):
        assert len(pred_hyps) == len(dataset), 'Number of predictions %d is not equal to the number of references %d' % (len(pred_hyps), len(dataset))
        pred_asts = [hyp[0].tree for hyp in pred_hyps]
        ref_asts = [ex.ast for ex in dataset]
        tables = [ex.db for ex in dataset]
        pred_sqls, ref_sqls = [], []
        for pred, ref, t in zip(pred_asts, ref_asts, tables):
            pred_sql = self.transition_system.ast_to_surface_code(pred, t)
            ref_sql = self.transition_system.ast_to_surface_code(ref, t)
            pred_sqls.append(pred_sql)
            ref_sqls.append(ref_sql)
        with tempfile.NamedTemporaryFile('w+t', encoding='utf8', suffix='.sql') as tmp_pred, \
                tempfile.NamedTemporaryFile('w+t', encoding='utf8', suffix='.sql') as tmp_ref, \
                    open(output_path, 'w', encoding='utf8') as of:
            for s in pred_sqls:
                tmp_pred.write(s + '\n')
            tmp_pred.flush()
            for s, t in zip(ref_sqls, tables):
                tmp_ref.write(s + '\t' + t['db_id'] + '\n')
            tmp_ref.flush()
            old_print = sys.stdout
            sys.stdout = of
            result_type = 'exact' if etype == 'match' else 'exec'
            all_exact_acc = evaluate(tmp_ref.name, tmp_pred.name, self.database_dir, etype, self.kmaps)['all'][result_type]
            sys.stdout = old_print
        return all_exact_acc

    def sql_acc(self, pred_hyps, dataset, output_path, etype, choice):
        """
            Specify the target dataset~(choice), `dev` or `train`
        """
        pred_sqls = []
        dbs = eval('self.' + choice + '_dbs')
        gold_file = eval('self.' + choice + '_gold_file')
        assert len(pred_hyps) == len(dbs), 'Number of predictions %d is not equal to the number of references %d' % (len(pred_hyps), len(dbs))
        for idx, hyp in enumerate(pred_hyps):
            best_ast = hyp[0].tree # by default, the top beam prediction
            pred_sql = self.transition_system.ast_to_surface_code(best_ast, self.tables[dbs[idx]])
            pred_sqls.append(pred_sql)
        with tempfile.NamedTemporaryFile('w+t', encoding='utf8', suffix='.sql') as tmp, open(output_path, 'w', encoding='utf8') as of:
            for s in pred_sqls:
                tmp.write(s + '\n')
            tmp.flush()
            old_print = sys.stdout
            sys.stdout = of
            result_type = 'exact' if etype == 'match' else 'exec'
            all_exact_acc = evaluate(gold_file, tmp.name, self.database_dir, etype, self.kmaps)['all'][result_type]
            sys.stdout = old_print
        return all_exact_acc

    def fscore(self, tuple_list, dataset, output_path, only_error=True, return_metric='acc'):
        """
        @input:
            tuple_list(list of tuple): each tuple contains
                pred_mask(list): batch_size x [max(tab_len) + max(col_len)]
                golden_mask(list): the same shape as pred_mask
                table_reverse_mappings and column_reverse_mappings: used to map position id to db schema id
                table_lens(list): batch_size
                column_lens(list): batch_size
            return_metric(str): acc, fscore, flag
        @output:
            precision, recall, fscore
        """
        assert return_metric in ['flag', 'acc', 'fscore']
        pred_results, reverse_mappings, golden_results = [], [], []
        for pm, gm, trm, crm, tl, cl in tuple_list:
            bias = max(tl)
            pm = [(e_pm[:tl[e_id]], e_pm[bias: bias + cl[e_id]]) for e_id, e_pm in enumerate(pm)]
            gm = [(e_gm[:tl[e_id]], e_gm[bias: bias + cl[e_id]]) for e_id, e_gm in enumerate(gm)]
            rm = [(t, c)for t, c in zip(trm, crm)]
            pred_results.extend(pm)
            golden_results.extend(gm)
            reverse_mappings.extend(rm)
        t_pm, c_pm = zip(*pred_results)
        t_gm, c_gm = zip(*golden_results)
        results = {'table': [], 'column': []}
        for e_id in range(len(t_pm)):
            pred_table_mask, ref_table_mask = t_pm[e_id], t_gm[e_id]
            results['table'].append(calculate_prfacc(pred_table_mask, ref_table_mask))
            pred_column_mask, ref_column_mask = c_pm[e_id], c_gm[e_id]
            results['column'].append(calculate_prfacc(pred_column_mask, ref_column_mask))
        acc, recall_acc = [], []
        with open(output_path, 'w', encoding='utf8') as of:
            for e_id in range(len(dataset)):
                if results['table'][e_id][1] == 1.0 and results['column'][e_id][1] == 1.0:
                    recall_acc.append(1)
                else:
                    recall_acc.append(0)
                if results['table'][e_id][2] == 1.0 and results['column'][e_id][2] == 1.0:
                    acc.append(1)
                else:
                    acc.append(0)
                if only_error and int(acc[-1]) == 1:
                    continue
                of.write('Database: %s\n' % (dataset[e_id].ex['db_id']))
                of.write('Question: %s\n' % (dataset[e_id].ex['question']))
                of.write('Query: %s\n' % (dataset[e_id].ex['query']))
                of.write('Used tables: %s\n' % (lexicalize_schema_items(t_gm[e_id], reverse_mappings[e_id][0], dataset[e_id].db, 'table')))
                of.write('Predict tables: %s\n' % (lexicalize_schema_items(t_pm[e_id], reverse_mappings[e_id][0], dataset[e_id].db, 'table')))
                of.write('Table Precision/Recall/Fscore/Acc: %s\n' % (results['table'][e_id]))
                of.write('Used columns: %s\n' % (lexicalize_schema_items(c_gm[e_id], reverse_mappings[e_id][1], dataset[e_id].db, 'column')))
                of.write('Predict columns: %s\n' % (lexicalize_schema_items(c_pm[e_id], reverse_mappings[e_id][1], dataset[e_id].db, 'column')))
                of.write('Column Precision/Recall/Fscore/Acc: %s\n\n' % (results['column'][e_id]))

            table_score, column_score = np.mean(results['table'], axis=0), np.mean(results['column'], axis=0)
            of.write('Overall Table Precision/Recall/Fscore/Acc: %s\n' % (table_score))
            of.write('Overall Column Precision/Recall/Fscore/Acc: %s\n' % (column_score))

            wrong_num, recall_wrong_num = len(dataset) - sum(acc), len(dataset) - sum(recall_acc)
            wrong_ratio, recall_wrong_ratio = float(wrong_num) / len(dataset), float(recall_wrong_num) / len(dataset)
            of.write('Total wrong samples number/ratio: %d/%.4f\n' % (wrong_num, wrong_ratio))
            of.write('Total recall wrong samples number/ratio: %d/%.4f' % (recall_wrong_num, recall_wrong_ratio))

        if return_metric == 'flag':
            return acc, recall_acc
        elif return_metric == 'acc':
            return 1 - wrong_ratio, 1 - recall_wrong_ratio
        else:
            return table_score[2], column_score[2]

def lexicalize_schema_items(select_mask, reverse_mapping, database, schema='table'):
    schema_items = []
    ids = [i for i, flag in enumerate(select_mask) if int(flag) == 1]
    if schema.lower() == 'table':
        names = database['table_names_original']
        schema_items = ['Table[%s]' % (names[reverse_mapping[idx]] if reverse_mapping[idx] < len(names) else 'other_db') for idx in ids]
    else:
        table_names = database['table_names_original']
        column_names = database['column_names_original']
        schema_ids = [column_names[reverse_mapping[idx]] if reverse_mapping[idx] < len(column_names) else (-1, 'other_db') for idx in ids]
        schema_items = ['Column[%s]' % (table_names[tab_id] + '.' + col_name if tab_id != -1 else col_name) for tab_id, col_name in schema_ids]
    return ', '.join(schema_items)

def calculate_prfacc(pred, ref):
    assert len(pred) == len(ref)
    num = float(len(pred))
    correct_num = sum(map(lambda x, y: 1 if x == y == 1 else 0, pred, ref))
    missing_num = sum(map(lambda x, y: 1 if x == 0 and y == 1 else 0, pred, ref))
    wrong_num = sum(map(lambda x, y: 1 if x == 1 and y == 0 else 0, pred, ref))
    p = float(correct_num) / (correct_num + wrong_num) if correct_num + wrong_num > 0 else 0.
    r = float(correct_num) / (correct_num + missing_num) if correct_num + missing_num > 0 else 0.
    f = 2 * p * r / (p + r) if p + r > 0 else 0.
    acc = sum(map(lambda x, y: 1 if x == y else 0, pred, ref)) / num
    return [p, r, f, acc]
