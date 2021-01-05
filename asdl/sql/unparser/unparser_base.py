#coding=utf8

from asdl.asdl import ASDLGrammar, ASDLConstructor, ASDLProduction
from asdl.asdl_ast import RealizedField, AbstractSyntaxTree

class UnParser():

    def __init__(self, grammar: ASDLGrammar):
        """ ASDLGrammar """
        super(UnParser, self).__init__()
        self.grammar = grammar

    @classmethod
    def from_grammar(cls, grammar: ASDLGrammar):
        grammar_name = grammar._grammar_name
        if 'v1' in grammar_name:
            from asdl.sql.unparser.unparser_v1 import UnParserV1
            return UnParserV1(grammar)
        elif 'v2' in grammar_name:
            from asdl.sql.unparser.unparser_v2 import UnParserV2
            return UnParserV2(grammar)
        else:
            raise ValueError('Not recognized grammar name %s' % (grammar_name))

    def unparse(self, sql_ast: AbstractSyntaxTree, db: dict, *args, **kargs):
        try:
            sql = self.unparse_sql(sql_ast, db, *args, **kargs)
            sql = ' '.join([i for i in sql.split(' ') if i != ''])
            return sql
        except Exception as e:
            print('Something Error happened while unparsing:', e)
            return 'SELECT * FROM %s' % (db['table_names_original'][0])

    def unparse_sql(self, sql_ast: AbstractSyntaxTree, db: dict, *args, **kargs):
        prod_name = sql_ast.production.constructor.name
        if prod_name == 'Intersect':
            return '%s INTERSECT %s' % (self.unparse_sql_unit(sql_ast.fields[0], db, *args, **kargs), self.unparse_sql_unit(sql_ast.fields[1], db, *args, **kargs))
        elif prod_name == 'Union':
            return '%s UNION %s' % (self.unparse_sql_unit(sql_ast.fields[0], db, *args, **kargs), self.unparse_sql_unit(sql_ast.fields[1], db, *args, **kargs))
        elif prod_name == 'Except':
            return '%s EXCEPT %s' % (self.unparse_sql_unit(sql_ast.fields[0], db, *args, **kargs), self.unparse_sql_unit(sql_ast.fields[1], db, *args, **kargs))
        else:
            return self.unparse_sql_unit(sql_ast.fields[0], db, *args, **kargs)

    def unparse_sql_unit(self, sql_field: RealizedField, db: dict, *args, **kargs):
        sql_ast = sql_field.value
        prod_name = sql_ast.production.constructor.name
        from_str = self.unparse_from(sql_ast.fields[0], db, *args, **kargs)
        select_str = self.unparse_select(sql_ast.fields[1], db, *args, **kargs)
        if prod_name == 'Complete':
            return 'SELECT %s FROM %s WHERE %s GROUP BY %s ORDER BY %s' % (
                select_str, from_str,
                self.unparse_where(sql_ast.fields[2], db, *args, **kargs),
                self.unparse_groupby(sql_ast.fields[3], db, *args, **kargs),
                self.unparse_orderby(sql_ast.fields[4], db, *args, **kargs)
            )
        elif prod_name == 'NoWhere':
            return 'SELECT %s FROM %s GROUP BY %s ORDER BY %s' % (
                select_str, from_str,
                self.unparse_groupby(sql_ast.fields[2], db, *args, **kargs),
                self.unparse_orderby(sql_ast.fields[3], db, *args, **kargs),
            )
        elif prod_name == 'NoGroupBy':
            return 'SELECT %s FROM %s WHERE %s ORDER BY %s' % (
                select_str, from_str,
                self.unparse_where(sql_ast.fields[2], db, *args, **kargs),
                self.unparse_orderby(sql_ast.fields[3], db, *args, **kargs),
            )
        elif prod_name == 'NoOrderBy':
            return 'SELECT %s FROM %s WHERE %s GROUP BY %s' % (
                select_str, from_str,
                self.unparse_where(sql_ast.fields[2], db, *args, **kargs),
                self.unparse_groupby(sql_ast.fields[3], db, *args, **kargs),
            )
        elif prod_name == 'OnlyWhere':
            return 'SELECT %s FROM %s WHERE %s' % (
                select_str, from_str,
                self.unparse_where(sql_ast.fields[2], db, *args, **kargs)
            )
        elif prod_name == 'OnlyGroupBy':
            return 'SELECT %s FROM %s GROUP BY %s' % (
                select_str, from_str,
                self.unparse_groupby(sql_ast.fields[2], db, *args, **kargs)
            )
        elif prod_name == 'OnlyOrderBy':
            return 'SELECT %s FROM %s ORDER BY %s' % (
                select_str, from_str,
                self.unparse_orderby(sql_ast.fields[2], db, *args, **kargs)
            )
        else:
            return 'SELECT %s FROM %s' % (select_str, from_str)

    def unparse_select(self, select_field: RealizedField, db: dict, *args, **kargs):
        raise NotImplementedError

    def unparse_from(self, from_field: RealizedField, db: dict, *args, **kargs):
        raise NotImplementedError

    def unparse_where(self, where_field: RealizedField, db: dict, *args, **kargs):
        return self.unparse_conds(where_field.value, db, *args, **kargs)

    def unparse_groupby(self, groupby_field: RealizedField, db: dict, *args, **kargs):
        raise NotImplementedError

    def unparse_orderby(self, orderby_field: RealizedField, db: dict, *args, **kargs):
        raise NotImplementedError

    def unparse_conds(self, conds_ast: AbstractSyntaxTree, db: dict, *args, **kargs):
        ctr_name = conds_ast.production.constructor.name
        if ctr_name in ['And', 'Or']:
            left_cond, right_cond = conds_ast.fields
            return self.unparse_conds(left_cond.value, db, *args, **kargs) + ' ' + ctr_name.upper() + ' ' + \
            self.unparse_conds(right_cond.value, db, *args, **kargs)
        else:
            return self.unparse_cond(conds_ast, db, *args, **kargs)
    
    def unparse_cond(self, cond_ast: AbstractSyntaxTree, db: dict, *args, **kargs):
        raise NotImplementedError
