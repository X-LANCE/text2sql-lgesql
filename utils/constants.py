GRAMMAR_FILEPATH = 'asdl/sql/grammar/sql_asdl_v2.txt'
COLUMN_TYPES = ['others', 'text', 'time', 'number', 'boolean']
SELF_RELATIONS = ['question-question-dist0', 'table-table-identity', 'column-column-identity']
# some other metapath-based relations ignored: column-column-sametable, table-table-fk, table-table-fkr, table-table-fkb
# r represents reverse edge, b represents bidirectional edge
MAX_RELATIVE_DIST = 2
RELATIONS = ['question-question-dist' + str(i) for i in range(- MAX_RELATIVE_DIST, 0)] + \
    ['question-question-dist' + str(i) for i in range(1, MAX_RELATIVE_DIST + 1)] + \
    ['column-column-fk', 'column-column-fkr'] + \
    ['table-column-pk', 'column-table-pk', 'table-column-has', 'column-table-has'] + \
    ['question-column-exactmatch', 'question-column-partialmatch', 'question-column-nomatch', 'question-column-valuematch',
    'column-question-exactmatch', 'column-question-partialmatch', 'column-question-nomatch', 'column-question-valuematch'] + \
    ['question-table-exactmatch', 'question-table-partialmatch', 'question-table-nomatch',
    'table-question-exactmatch', 'table-question-partialmatch', 'table-question-nomatch']