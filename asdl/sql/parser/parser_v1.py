#coding=utf8
from asdl.sql.parser.parser_base import Parser
from asdl.asdl import ASDLGrammar
from asdl.asdl_ast import RealizedField, AbstractSyntaxTree

class ParserV1(Parser):
    """ In this version, we eliminate all cardinality ? and restrict that * must have at least one item
    """
    def parse_select(self, select_clause: list, select_field: RealizedField):
        """
            ignore cases agg(col_id1 op col_id2) and agg(col_id1) op agg(col_id2)
        """
        select_clause = select_clause[1] # list of (agg, val_unit)
        unit_op_list = ['Unary', 'Minus', 'Plus', 'Times', 'Divide']
        for agg, val_unit in select_clause:
            if agg != 0: # agg col_id
                ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name('Unary'))
                agg_field, col_field = ast_node.fields
                agg_field.add_value(self.parse_agg(agg))
                col_field.add_value(int(val_unit[1][1]))
            else: # binary_op col_id1 col_id2
                ast_node = self.parse_val_unit(val_unit)
            select_field.add_value(ast_node)

    def parse_from(self, from_clause: dict, from_field: RealizedField):
        """
            Ignore from conditions, since it is not evaluated in evaluation script
        """
        table_units = from_clause['table_units']
        t = table_units[0][0]
        if t == 'table_unit':
            ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name('FromTable'))
            tables_field = ast_node.fields[0]
            for _, v in table_units:
                tables_field.add_value(int(v))
        else:
            assert t == 'sql'
            v = table_units[0][1]
            ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name('FromSQL'))
            ast_node.fields[0].add_value(self.parse_sql(v))
        from_field.add_value(ast_node)

    def parse_groupby(self, groupby_clause: list, having_clause: list, groupby_field: RealizedField):
        col_ids = []
        for col_unit in groupby_clause:
            col_ids.append(col_unit[1]) # agg is None and isDistinct False
        if having_clause:
            ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name('Having'))
            col_ids_field, having_fields = ast_node.fields
            having_fields.add_value(self.parse_conds(having_clause))
        else:
            ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name('NoHaving'))
            col_ids_field = ast_node.fields[0]
        for idx in col_ids:
            col_ids_field.add_value(int(idx))
        groupby_field.add_value(ast_node)

    def parse_orderby(self, orderby_clause: list, limit: int, orderby_field: RealizedField):
        if limit is None:
            ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name('Asc')) if orderby_clause[0] == 'asc' \
                else AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name('Desc'))
        else:
            ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name('AscLimit')) if orderby_clause[0] == 'asc' \
                else AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name('DescLimit'))
        val_units_field = ast_node.fields[0]
        for val_unit in orderby_clause[1]:
            val_units_field.add_value(self.parse_val_unit(val_unit))
        orderby_field.add_value(ast_node)

    def parse_cond(self, cond: list):
        not_op, cmp_op, val_unit, val1, val2 = cond
        not_op = '^' if not_op else ''
        op_list = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists')
        cmp_op = not_op + op_list[cmp_op]
        op_dict = {
            'between': 'Between', '=': 'Eq', '>': 'Gt', '<': 'Lt', '>=': 'Ge', '<=': 'Le', '!=': 'Neq',
            'in': 'In', 'like': 'Like', '^in': 'NotIn', '^like': 'NotLike'
        }
        ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name(op_dict[cmp_op]))
        val_unit_field, *val_field = ast_node.fields
        val_unit_field.add_value(self.parse_val_unit(val_unit))
        val_field[0].add_value(self.parse_val(val1))
        if val2:
            val_field[1].add_value(self.parse_val(val2))
        return ast_node

    def parse_val_unit(self, val_unit: list):
        unit_op, col_unit1, col_unit2 = val_unit
        unit_op_list = ['Unary', 'Minus', 'Plus', 'Times', 'Divide']
        ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name(unit_op_list[unit_op]))
        if unit_op == 0:
            agg, col_id, _ = col_unit1
            ast_node.fields[0].add_value(self.parse_agg(agg))
            ast_node.fields[1].add_value(int(col_id))
        else:
            col_id1, col_id2 = col_unit1[1], col_unit2[1]
            ast_node.fields[0].add_value(int(col_id1))
            ast_node.fields[1].add_value(int(col_id2))
        return ast_node

    def parse_val(self, val):
        if type(val) == list:
            # col_unit
            ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name('ColumnVal'))
            ast_node.fields[0].add_value(int(val[1]))
            return ast_node
        elif type(val) == dict:
            # nested sql
            ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name('SQLVal'))
            ast_node.fields[0].add_value(self.parse_sql(val))
            return ast_node
        else:
            # float or string or int
            return AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name('Value'))

    def parse_agg(self, agg: int):
        agg_mapping = ['None', 'Max', 'Min', 'Count', 'Sum', 'Avg']
        return AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name(agg_mapping[agg]))
