#coding=utf8
from asdl.sql.unparser.unparser_base import UnParser
from asdl.asdl import ASDLGrammar, ASDLConstructor, ASDLProduction
from asdl.asdl_ast import RealizedField, AbstractSyntaxTree

class UnParserV1(UnParser):

    def unparse_select(self, select_field: RealizedField, db: dict, *args, **kargs):
        select_list = select_field.value
        select_items = []
        for val_unit_ast in select_list:
            val_unit_str = self.unparse_val_unit(val_unit_ast, db, *args, **kargs)
            select_items.append(val_unit_str)
        return ' , '.join(select_items)

    def unparse_from(self, from_field: RealizedField, db: dict, *args, **kargs):
        from_ast = from_field.value
        ctr_name = from_ast.production.constructor.name
        if ctr_name == 'FromTable':
            tab_ids = from_ast.fields[0].value
            if len(tab_ids) == 1:
                return db['table_names_original'][tab_ids[0]]
            else:
                tab_names = [db['table_names_original'][i] for i in tab_ids]
                return ' JOIN '.join(tab_names)
        else:
            sql_ast = from_ast.fields[0].value
            return '( ' + self.unparse_sql(sql_ast, db, *args, **kargs) + ' )'

    def unparse_groupby(self, groupby_field: RealizedField, db: dict, *args, **kargs):
        groupby_ast = groupby_field.value
        ctr_name = groupby_ast.production.constructor.name
        col_ids = groupby_ast.fields[0]
        groupby_str = []
        for col_id in col_ids.value:
            if int(col_id) != 0:
                tab_id, col_name = db['column_names_original'][int(col_id)]
                tab_name = db['table_names_original'][tab_id]
                groupby_str.append(tab_name + '.' + col_name)
            else:
                groupby_str.append('*')
        groupby_str = ' , '.join(groupby_str)
        if ctr_name == 'Having':
            having = groupby_ast.fields[1].value
            having_str = self.unparse_conds(having, db, *args, **kargs)
            return groupby_str + ' HAVING ' + having_str 
        else:
            return groupby_str

    def unparse_orderby(self, orderby_field: RealizedField, db: dict, *args, **kargs):
        orderby_ast = orderby_field.value
        ctr_name = orderby_ast.production.constructor.name.lower()
        val_unit_ast_list = orderby_ast.fields[0].value
        val_unit_str = []
        for val_unit_ast in val_unit_ast_list:
            val_unit_str.append(self.unparse_val_unit(val_unit_ast, db, *args, **kargs))
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
        op_dict = {
            'Between': ' BETWEEN ', 'Eq': ' = ', 'Gt': ' > ', 'Lt': ' < ', 'Ge': ' >= ', 'Le': ' <= ', 'Neq': ' != ',
            'In': ' IN ', 'Like': ' LIKE ', 'NotIn': ' NOT IN ', 'NotLike': ' NOT LIKE ' 
        }
        if ctr_name == 'Between':
            val_unit_str = self.unparse_val_unit(cond_ast.fields[0].value, db, *args, **kargs)
            val1_str = self.unparse_val(cond_ast.fields[1].value, db, *args, **kargs)
            val2_str = self.unparse_val(cond_ast.fields[2].value, db, *args, **kargs)
            return val_unit_str + ' BETWEEN ' + val1_str + ' AND ' + val2_str
        else:
            op = op_dict[ctr_name]
            val_unit_str = self.unparse_val_unit(cond_ast.fields[0].value, db, *args, **kargs)
            val_str = self.unparse_val(cond_ast.fields[1].value, db, *args, **kargs)
            return op.join([val_unit_str, val_str])
    
    def unparse_val_unit(self, val_unit_ast: AbstractSyntaxTree, db: dict, *args, **kargs):
        unit_op = val_unit_ast.production.constructor.name
        if unit_op == 'Unary':
            agg = self.unparse_agg(val_unit_ast.fields[0].value)
            col_id = int(val_unit_ast.fields[1].value)
            if col_id != 0:
                tab_id, col_name = db['column_names_original'][col_id]
                tab_name = db['table_names_original'][tab_id]
                schema_item = tab_name + '.' + col_name
            else:
                schema_item = '*'
            schema_item = schema_item if agg.lower() == 'none' else agg + '(' + schema_item + ')'
            return schema_item
        else:
            binary = {'Minus': ' - ', 'Plus': ' + ', 'Times': ' * ', 'Divide': ' / '}
            op = binary[unit_op]
            col_id1, col_id2 = int(val_unit_ast.fields[0].value), int(val_unit_ast.fields[1].value)
            if col_id1 != 0:
                tab_id1, col_name1 = db['column_names_original'][col_id1]
                tab_name1 = db['table_names_original'][tab_id1]
                schema_item1 = tab_name1 + '.' + col_name1
            else:
                schema_item1 = '*'
            if col_id2 != 0:
                tab_id2, col_name2 = db['column_names_original'][col_id2]
                tab_name2 = db['table_names_original'][tab_id2]
                schema_item2 = tab_name2 + '.' + col_name2
            else:
                schema_item2 = '*'
            return op.join([schema_item1, schema_item2])

    def unparse_val(self, val_ast: AbstractSyntaxTree, db: dict, *args, **kargs):
        val_ctr_name = val_ast.production.constructor.name
        if val_ctr_name == 'ColumnVal':
            col_id = int(val_ast.fields[0].value)
            if col_id == 0:
                return '*'
            else:
                tab_id, col_name = db['column_names_original'][col_id]
                tab_name = db['table_names_original'][tab_id]
                return tab_name + '.' + col_name
        elif val_ctr_name == 'SQLVal':
            return '( ' + self.unparse_sql(val_ast.fields[0].value, db, *args, **kargs) + ' )'
        else:
            return '"value"'
 
    def unparse_agg(self, agg_ast: AbstractSyntaxTree):
        return agg_ast.production.constructor.name.upper() # None, Max, Min, Count, Sum, Avg