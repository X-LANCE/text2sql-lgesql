#coding=utf8
import sys, tempfile, os
import numpy as np
from asdl.sql.sql_transition_system import SelectColumnAction, SelectTableAction
from evaluation import evaluate, build_foreign_key_map_from_json, build_valid_col_units, rebuild_sql_val, rebuild_sql_col, eval_exec_match
from evaluation import Evaluator as Engine
from process_sql import get_schema, Schema, get_sql

class Evaluator():

    def __init__(self, transition_system, table_path='data/tables.json', database_dir='data/database'):
        super(Evaluator, self).__init__()
        self.transition_system = transition_system
        self.kmaps = build_foreign_key_map_from_json(table_path)
        self.database_dir = database_dir
        self.engine = Engine()
        self.acc_dict = {
            "sql": self.sql_acc, # use golden sql as references
            "ast": self.ast_acc, # compare ast accuracy, ast may be incorrect when constructed from raw sql
            "beam": self.beam_acc, # if the correct answer exist in the beam, assume the result is true
        }

    def acc(self, pred_hyps, dataset, output_path=None, acc_type='sql', etype='match'):
        assert len(pred_hyps) == len(dataset) and acc_type in self.acc_dict and etype in ['match', 'exec']
        acc_method = self.acc_dict[acc_type]
        return acc_method(pred_hyps, dataset, output_path, etype)

    def beam_acc(self, pred_hyps, dataset, output_path, etype):
        scores, results = {}, []
        for each in ['easy', 'medium', 'hard', 'extra', 'all']:
            scores[each] = [0, 0.] # first is count, second is total score
        for idx, pred in enumerate(pred_hyps):
            question, gold_sql, db = dataset[idx].ex['question'], dataset[idx].ex['query'], dataset[idx].db
            for b_id, hyp in enumerate(pred):
                pred_sql = self.transition_system.ast_to_surface_code(hyp.tree, db)
                score, hardness = self.single_acc(pred_sql, gold_sql, db['db_id'], etype)
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
                pred_sql = self.transition_system.ast_to_surface_code(pred[0].tree, db)
                results.append((hardness, question, gold_sql, pred_sql, 0, False))
        for each in scores:
            accuracy = scores[each][1] / float(scores[each][0]) if scores[each][0] != 0 else 0.
            scores[each].append(accuracy)
        of = open(output_path, 'w', encoding='utf8') if output_path is not None else \
            tempfile.TemporaryFile('w+t')
        for item in results:
            of.write('Level: %s\n' % (item[0]))
            of.write('Question: %s\n' % (item[1]))
            of.write('Gold SQL: %s\n' %(item[2]))
            of.write('Pred SQL (%s): %s\n' % (item[4], item[3]))
            of.write('Correct: %s\n\n' % (item[5]))
        for each in scores:
            of.write('Level %s: %s\n' % (each, scores[each]))
        of.close()
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

    def ast_acc(self, pred_hyps, dataset, output_path, etype):
        pred_asts = [hyp[0].tree for hyp in pred_hyps]
        ref_asts = [ex.ast for ex in dataset]
        dbs = [ex.db for ex in dataset]
        pred_sqls, ref_sqls = [], []
        for pred, ref, db in zip(pred_asts, ref_asts, dbs):
            pred_sql = self.transition_system.ast_to_surface_code(pred, db)
            ref_sql = self.transition_system.ast_to_surface_code(ref, db)
            pred_sqls.append(pred_sql)
            ref_sqls.append(ref_sql)
        with tempfile.NamedTemporaryFile('w+t', encoding='utf8', suffix='.sql') as tmp_pred, \
                tempfile.NamedTemporaryFile('w+t', encoding='utf8', suffix='.sql') as tmp_ref:
            of = open(output_path, 'w', encoding='utf8') if output_path is not None \
                else tempfile.TemporaryFile('w+t')
            # write pred and ref sqls
            for s in pred_sqls:
                tmp_pred.write(s + '\n')
            tmp_pred.flush()
            for s, db in zip(ref_sqls, dbs):
                tmp_ref.write(s + '\t' + db['db_id'] + '\n')
            tmp_ref.flush()
            # calculate ast accuracy
            old_print = sys.stdout
            sys.stdout = of
            result_type = 'exact' if etype == 'match' else 'exec'
            all_exact_acc = evaluate(tmp_ref.name, tmp_pred.name, self.database_dir, etype, self.kmaps)['all'][result_type]
            sys.stdout = old_print
            of.close()
        return float(all_exact_acc)

    def sql_acc(self, pred_hyps, dataset, output_path, etype):
        pred_sqls, ref_sqls = [], [ex['query'] for ex in dataset]
        dbs = [ex.db for ex in dataset]
        for idx, hyp in enumerate(pred_hyps):
            best_ast = hyp[0].tree # by default, the top beam prediction
            pred_sql = self.transition_system.ast_to_surface_code(best_ast, dbs[idx])
            pred_sqls.append(pred_sql)
        with tempfile.NamedTemporaryFile('w+t', encoding='utf8', suffix='.sql') as tmp_pred, \
            tempfile.NamedTemporaryFile('w+t', encoding='utf8', suffix='.sql') as tmp_ref:
            of = open(output_path, 'w', encoding='utf8') if output_path is not None \
                else tempfile.TemporaryFile('w+t')
            # write pred and ref sqls
            for s in pred_sqls:
                tmp.write(s + '\n')
            tmp.flush()
            for s, db in zip(ref_sqls, dbs):
                tmp_ref.write(s + '\t' + db['db_id'] + '\n')
            tmp_ref.flush()
            # calculate sql accuracy
            old_print = sys.stdout
            sys.stdout = of
            result_type = 'exact' if etype == 'match' else 'exec'
            all_exact_acc = evaluate(tmp_ref.name, tmp_pred.name, self.database_dir, etype, self.kmaps)['all'][result_type]
            sys.stdout = old_print
            of.close()
        return float(all_exact_acc)