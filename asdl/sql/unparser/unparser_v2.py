#coding=utf8
from asdl.sql.unparser.unparser_base import UnParser
from asdl.asdl import ASDLGrammar, ASDLConstructor, ASDLProduction
from asdl.asdl_ast import RealizedField, AbstractSyntaxTree

class UnParserV2(UnParser):

    def unparse_select(self, select_field: RealizedField, db: dict, *args, **kargs):
        select_ast = select_field.value
        select_list = select_ast.fields
        select_items = []
        for val_unit_field in select_list:
            val_unit_str = self.unparse_val_unit(val_unit_field.value, db, *args, **kargs)
            select_items.append(val_unit_str)
        return ' , '.join(select_items)

    def unparse_from(self, from_field: RealizedField, db: dict, *args, **kargs):
        from_ast = from_field.value
        ctr_name = from_ast.production.constructor.name
        if 'Table' in ctr_name:
            tab_names = []
            for tab_field in from_ast.fields:
                tab_name = db['table_names_original'][int(tab_field.value)]
                tab_names.append(tab_name)
            return ' JOIN '.join(tab_names)
        else:
            return '( ' + self.unparse_sql(from_ast.fields[0].value, db, *args, **kargs) + ' )'

    def unparse_groupby(self, groupby_field: RealizedField, db: dict, *args, **kargs):
        groupby_ast = groupby_field.value
        ctr_name = groupby_ast.production.constructor.name
        groupby_str = []
        num = len(groupby_ast.fields) if 'NoHaving' in ctr_name else len(groupby_ast.fields) - 1
        for col_id_field in groupby_ast.fields[:num]:
            # col_id = int(col_id_field.value)
            # tab_id, col_name = db['column_names_original'][col_id]
            # if col_id != 0:
                # tab_name = db['table_names_original'][tab_id]
                # col_name = tab_name + '.' + col_name
            col_name = self.unparse_col_unit(col_id_field.value, db, *args, **kargs)
            groupby_str.append(col_name)
        groupby_str = ' , '.join(groupby_str)
        if 'NoHaving' in ctr_name:
            return groupby_str
        else:
            having = groupby_ast.fields[-1].value
            having_str = self.unparse_conds(having, db, *args, **kargs)
            return groupby_str + ' HAVING ' + having_str

    def unparse_orderby(self, orderby_field: RealizedField, db: dict, *args, **kargs):
        orderby_ast = orderby_field.value
        ctr_name = orderby_ast.production.constructor.name.lower()
        val_unit_str = []
        for val_unit_field in orderby_ast.fields:
            val_unit_ast = val_unit_field.value
            val_unit_str.append(self.unparse_col_unit(val_unit_ast, db, *args, **kargs))
            # val_unit_str.append(self.unparse_val_unit(val_unit_ast, db, *args, **kargs))
        val_unit_str = ' , '.join(val_unit_str)
        if 'asc' in ctr_name and 'limit' in ctr_name:
            return '%s ASC LIMIT 1' % (val_unit_str)
        elif 'asc' in ctr_name:
            return '%s ASC' % (val_unit_str)
        elif 'desc' in ctr_name and 'limit' in ctr_name:
            return '%s DESC LIMIT 1' % (val_unit_str)
        else:
            return '%s DESC' % (val_unit_str)

    def unparse_cond(self, cond_ast: AbstractSyntaxTree, db: dict, *args, **kargs):
        ctr_name = cond_ast.production.constructor.name
        val_unit_str = self.unparse_val_unit(cond_ast.fields[0].value, db, *args, **kargs)
        val_str =  '( ' + self.unparse_sql(cond_ast.fields[1].value, db, *args, **kargs) + ' )' if len(cond_ast.fields) == 2 else '"value"'
        if ctr_name.startswith('Between'):
            return val_unit_str + ' BETWEEN ' + val_str + ' AND "value"'
        else:
            op_dict = {
                'Between': ' BETWEEN ', 'Eq': ' = ', 'Gt': ' > ', 'Lt': ' < ', 'Ge': ' >= ', 'Le': ' <= ', 'Neq': ' != ',
                'In': ' IN ', 'Like': ' LIKE ', 'NotIn': ' NOT IN ', 'NotLike': ' NOT LIKE '
            }
            ctr_name = ctr_name if 'SQL' not in ctr_name else ctr_name[:ctr_name.index('SQL')]
            op = op_dict[ctr_name]
            return op.join([val_unit_str, val_str])

    def unparse_val_unit(self, val_unit_ast: AbstractSyntaxTree, db: dict, *args, **kargs):
        unit_op = val_unit_ast.production.constructor.name
        if unit_op == 'Unary':
            return self.unparse_col_unit(val_unit_ast.fields[0].value, db, *args, **kargs)
        else:
            binary = {'Minus': ' - ', 'Plus': ' + ', 'Times': ' * ', 'Divide': ' / '}
            op = binary[unit_op]
            return op.join([self.unparse_col_unit(val_unit_ast.fields[0].value, db, *args, **kargs),
                self.unparse_col_unit(val_unit_ast.fields[1].value, db, *args, **kargs)])
            # col_id1, col_id2 = int(val_unit_ast.fields[0].value), int(val_unit_ast.fields[1].value)
            # tab_id1, col_name1 = db['column_names_original'][col_id1]
            # if col_id1 != 0:
                # tab_name1 = db['table_names_original'][tab_id1]
                # col_name1 = tab_name1 + '.' + col_name1
            # tab_id2, col_name2 = db['column_names_original'][col_id2]
            # if col_id2 != 0:
                # tab_name2 = db['table_names_original'][tab_id2]
                # col_name2 = tab_name2 + '.' + col_name2
            # return op.join([col_name1, col_name2])

    def unparse_col_unit(self, col_unit_ast: AbstractSyntaxTree, db: dict, *args, **kargs):
        agg = col_unit_ast.production.constructor.name
        col_id = int(col_unit_ast.fields[0].value)
        tab_id, col_name = db['column_names_original'][col_id]
        if col_id != 0:
            tab_name = db['table_names_original'][tab_id]
            col_name = tab_name + '.' + col_name
        if agg == 'None':
            return col_name
        else: # Max/Min/Count/Sum/Avg
            return agg.upper() + '(' + col_name + ')'
