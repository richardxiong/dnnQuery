'''
The following functions parse the logical forms
@args:
table: a list of dictionaries of the table corresponding to the logical form
logicalForm: a list of key words with length as the hash key
if logical form not right, return ['N.A.']
.
specific function names are of the form parser_len(#LOGICALFORM LENGTH)_(#TYPE OF THIS LENGTH)
wrapper function names are of the form parser_len(#LOGICALFORM LENGTH)

(NOTE)
1. since there might be multiple satisified answer to one logical form, the return type is list
2. all numerical values are of type float in the process, and string in the output
'''
NOTFOUND = []
compPool = ['equal', 'less', 'greater']
argPool = ['argmax', 'argmin']
ERROR = 1e-6
'''
helper function, check whether is a number
'''
def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def checkCompare(comp, val1, val2):
    if comp == 'equal' and abs(val1 - val2) < ERROR:
        return True
    if comp == 'less' and val1 < val2 - ERROR:
        return True
    if comp == 'greater' and val1 > val2 + ERROR:
        return True

    return False
'''
length = 1
(1) sum: number of rows in the table
'''
def parser_len1_type1(table, logicalForm):
    if logicalForm[0] == 'sum':
        return [str(len(table))]
    return NOTFOUND

def parser_len1(table, logicalForm):
    return parser_len1_type1(table, logicalForm)
'''
length = 2
(1) sum {query_field}: the summation of values in the given field
(2) mean {query_field}: the mean of values in the given field
'''
def parser_len2_type1(table, logicalForm):
    query_field = logicalForm[1]
    result = 0
    for i in range(len(table)):
        row = table[i]
        if query_field not in row.keys() or isfloat(row[query_field]) == False:
            return NOTFOUND
        result += float(row[query_field])
    return [str(result)]

def parser_len2_type2(table, logicalForm):
    query_field = logicalForm[1]
    result = 0
    for i in range(len(table)):
        row = table[i]
        if query_field not in row.keys() or isfloat(row[query_field]) == False:
            return NOTFOUND
        result += float(row[query_field])
    return [str(result / len(table))]

def parser_len2(table, logicalForm):
    if logicalForm[0] == 'sum':
        return parser_len2_type1(table, logicalForm)
    elif logicalForm[0] == 'mean':
        return parser_len2_type2(table, logicalForm)
    else:
        return NOTFOUND

'''
length = 3
(1) arg {max_min} {arg1} {arg2}
'''
def parser_len3_type1(table, logicalForm):
    argminmax, arg1, arg2 = logicalForm[0], logicalForm[1], logicalForm[2]
    if argminmax not in argPool:
        return NOTFOUND

    criteria = float('-inf') if argminmax == 'argmax' else float('inf')
    for i in range(len(table)):
        row = table[i]
        if arg2 not in row.keys() or arg1 not in row.keys() or isfloat(row[arg2]) == False:
            return NOTFOUND
        if argminmax == 'argmax' and float(row[arg2]) > criteria:
            criteria = float(row[arg2])
        if argminmax == 'argmin' and float(row[arg2]) < criteria:
            criteria = float(row[arg2])
    # in case of multiple results
    result = []
    for i in range(len(table)):
        row = table[i]
        if float(row[arg2]) == criteria:
            result.append(row[arg1])
    return result

def parser_len3(table, logicalForm):
    if logicalForm[0] in argPool:
        return parser_len3_type1(table, logicalForm)
    return NOTFOUND

'''
length = 5
(1) where {comp_field1} {comp} {comp_val} sum
'''
def parser_len5_type1(table, logicalForm):
    comp_field1, comp, comp_val = logicalForm[1], logicalForm[2], logicalForm[3]
    if comp not in compPool:
        return NOTFOUND

    if isfloat(comp_val) == False:
        return NOTFOUND
    queryTable = []
    for i in range(len(table)):
        row = table[i]
        if comp_field1 not in row.keys() or isfloat(row[comp_field1]) == False:
            return NOTFOUND
        if checkCompare(comp, float(row[comp_field1]), float(comp_val)):
            queryTable.append(row)
    queryLogical = ['sum']
    return parser_len1(queryTable, queryLogical)

def parser_len5(table, logicalForm):
    if logicalForm[0] == 'where' and logicalForm[-1] == 'sum':
        return parser_len5_type1(table, logicalForm)
    return NOTFOUND
'''
length = 6
(1) where {comp_field} {comp} {comp_val} select {select_field}
(2) where {comp_field1} {comp} {comp_val} sum {comp_field2}
(3) where {comp_field1} {comp} {comp_val} mean {comp_field2}
'''
def parser_len6_type1(table, logicalForm):
    comp_field, comp, comp_val, select_field = logicalForm[1], logicalForm[2], logicalForm[3], logicalForm[5]
    if comp not in compPool or isfloat(comp_val) == False:
        return NOTFOUND
    result = []
    for i in range(len(table)):
        row = table[i]
        if comp_field not in row.keys() or select_field not in row.keys() or isfloat(row[comp_field]) == False:
            return NOTFOUND
        if checkCompare(comp, float(row[comp_field]), float(comp_val)):
            result.append(row[select_field])
    return result

def parser_len6_type2(table, logicalForm):
    comp_field1, comp, comp_val, comp_field2 = logicalForm[1], logicalForm[2], logicalForm[3], logicalForm[5]
    if comp not in compPool or isfloat(comp_val) == False:
        return NOTFOUND
    queryTable = []
    for i in range(len(table)):
        row = table[i]
        if comp_field1 not in row.keys() or comp_field2 not in row.keys():
            return NOTFOUND
        if isfloat(row[comp_field1]) == False or isfloat(row[comp_field2]) == False:
            return NOTFOUND
        if checkCompare(comp, float(row[comp_field1]), float(comp_val)):
            queryTable.append(row)

    queryLogical = ['sum', comp_field2]

    return parser_len2(queryTable, queryLogical)

def parser_len6_type3(table, logicalForm):
    comp_field1, comp, comp_val, comp_field2 = logicalForm[1], logicalForm[2], logicalForm[3], logicalForm[5]
    if comp not in compPool or isfloat(comp_val) == False:
        return NOTFOUND
    queryTable = []

    for i in range(len(table)):
        row = table[i]
        if comp_field1 not in row.keys() or comp_field2 not in row.keys():
            return NOTFOUND
        if isfloat(row[comp_field1]) == False or isfloat(row[comp_field2]) == False:
            return NOTFOUND

        if checkCompare(comp, float(row[comp_field1]), float(comp_val)):
            queryTable.append(row)
    queryLogical = ['mean', comp_field2]
    return parser_len2(queryTable, queryLogical)

def parser_len6(table, logicalForm):
    if logicalForm[0] != 'where':
        return NOTFOUND
    if logicalForm[4] == 'select':
        return parser_len6_type1(table, logicalForm)
    elif logicalForm[4] == 'sum':
        return parser_len6_type2(table, logicalForm)
    elif logicalForm[4] == 'mean':
        return parser_len6_type3(table, logicalForm)
    return NOTFOUND

'''
length = 7
(1) where {comp_field} {comp} {comp_val} arg {max_min} {arg1} {arg2}
'''
def parser_len7_type1(table, logicalForm):
    comp_field, comp, comp_val, argminmax, arg1, arg2 = \
        logicalForm[1], logicalForm[2], logicalForm[3], logicalForm[4], logicalForm[5], logicalForm[6]

    if comp not in compPool:
        return NOTFOUND
    if argminmax not in argPool:
        return NOTFOUND
    if isfloat(comp_val) == False:
        return NOTFOUND
    queryTable = []
    for i in range(len(table)):
        row = table[i]
        if comp_field not in row.keys() or arg1 not in row.keys() or arg2 not in row.keys():
            return NOTFOUND
        if isfloat(row[comp_field]) == False or isfloat(row[arg2]) == False:
            return NOTFOUND
        if checkCompare(comp, float(row[comp_field]), float(comp_val)):
            queryTable.append(row)

    queryLogical = [argminmax, arg1, arg2]
    return parser_len3(queryTable ,queryLogical)

def parser_len7(table, logicalForm):
    if logicalForm[0] != 'where':
        return NOTFOUND
    return parser_len7_type1(table, logicalForm)
'''
length = 10
(1) where {comp_field1} {comp1} {comp_val1} where {comp_field2} {comp2} {comp_val2}
    select {select_field}
'''
def parser_len10_type1(table, logicalForm):
    comp_field1, comp1, comp_val1 = logicalForm[1], logicalForm[2], logicalForm[3]

    comp_field2, comp2, comp_val2 = logicalForm[5], logicalForm[6], logicalForm[7]

    select_field = logicalForm[9]

    queryLogical = ['where', comp_field1, comp1, comp_val1, 'select', select_field]
    result1 = parser_len6(table, queryLogical)
    queryLogical = ['where', comp_field2, comp2, comp_val2, 'select', select_field]
    result2 = parser_len6(table, queryLogical)
    select_field = logicalForm[9]

    result = []
    for item in result1:
        if item in result2:
            result.append(item)

    return result

def parser_len10(table, logicalForm):
    if logicalForm[0] == 'where' and logicalForm[4] == 'where' and logicalForm[8] == 'select':
        return parser_len10_type1(table, logicalForm)
    return NOTFOUND
'''
length = 11
(1) where {comp_field1} {comp1} {comp_val1} where {comp_field2} {comp1} {comp_val2}
    arg {max_min} {arg1} {arg2}
'''
def parser_len11_type1(table, logicalForm):
    comp_field1, comp1, comp_val1 = logicalForm[1], logicalForm[2], logicalForm[3]
    if comp1 not in compPool or isfloat(comp_val1) == False:
        return NOTFOUND
    comp_field2, comp2, comp_val2 = logicalForm[5], logicalForm[6], logicalForm[7]
    if comp2 not in compPool or isfloat(comp_val2) == False:
        return NOTFOUND

    argminmax, arg1, arg2 = logicalForm[8], logicalForm[9], logicalForm[10]

    rowIdx1 = []
    for i in range(len(table)):
        row = table[i]
        if comp_field1 not in row.keys():
            return NOTFOUND
        if checkCompare(comp1, float(row[comp_field1]), float(comp_val1)):
            rowIdx1.append(i)

    queryTable = []
    for i in rowIdx1:
        row = table[i]
        if comp_field2 not in row.keys() or arg1 not in row.keys() or arg2 not in row.keys():
            return NOTFOUND
        if checkCompare(comp2, float(row[comp_field2]), float(comp_val2)):
            queryTable.append(row)

    queryLogical = [argminmax, arg1, arg2]
    return parser_len3(queryTable, queryLogical)


def parser_len11(table, logicalForm):
    if logicalForm[0] == 'where' and logicalForm[4] == 'where':
        return parser_len11_type1(table, logicalForm)
    return NOTFOUND
'''
length = 13
(1) arg {max_min} {arg1} {arg2} as A arg {max_min} {arg1} {arg2} as B diff A B
'''
def parser_len13_type1(table, logicalForm):
    argminmax, arg1, arg2 = logicalForm[0], logicalForm[1], logicalForm[2]
    queryLogical = [argminmax, arg1, arg2]
    result1 = parser_len3(table, queryLogical)
    if result1 == []:
        return NOTFOUND

    argminmax, arg1, arg2 = logicalForm[5], logicalForm[6], logicalForm[7]
    queryLogical = [argminmax, arg1, arg2]
    result2 = parser_len3(table, queryLogical)
    if result2 == []:
        return NOTFOUND

    result = []
    for val1 in result1:
        for val2 in result2:
            if isfloat(val1) == False or isfloat(val2) == False:
                continue
            result.append(str(float(val1) - float(val2)))

    return result

def parser_len13(table, logicalForm):
    if logicalForm[3] == 'as' and logicalForm[4] == 'A' and logicalForm[8] == 'as' \
        and logicalForm[9] == 'B' and logicalForm[10] == 'diff' and logicalForm[11] == 'A' and logicalForm[12] == 'B':
        return parser_len13_type1(table, logicalForm)

    return NOTFOUND
'''
length = 15
(1) where {query1_comp_field} {query1_comp} {query1_comp_val} select {query1_project_field} as A
    where {query1_project_field} {query2_comp} A arg {max_min} {arg1} {arg2}
'''
def parser_len15_type1(table, logicalForm):
    query1_comp_field, query1_comp, query1_comp_val = logicalForm[1], logicalForm[2], logicalForm[3]
    query1_project_field = logicalForm[5]
    queryLogical = ['where', query1_comp_field, query1_comp, query1_comp_val, 'select', query1_project_field]
    result1 = parser_len6(table, queryLogical)
    if result1 == [] or isfloat(result1[0]) == False:
        return NOTFOUND

    query2_comp = logicalForm[10]
    argminmax, arg1, arg2 = logicalForm[12], logicalForm[13], logicalForm[14]
    queryLogical = ['where', query1_project_field, query2_comp, result1[0], argminmax, arg1, arg2]

    return parser_len7(table, queryLogical)


def parser_len15(table, logicalForm):
    if logicalForm[0] == 'where' and logicalForm[4] == 'select' and logicalForm[6] == 'as' \
        and logicalForm[7] == 'A' and logicalForm[8] == 'where' and logicalForm[11] == 'A':
        return parser_len15_type1(table, logicalForm)
'''
length = 19
(1) where {query1_comp_field} {query1_comp} {query1_comp_val} select {query1_project_field} as A
    where {query2_comp_field} {query2_comp} {query2_comp_val} select {query2_project_field} as B
    sum A B
(2) where {query1_comp_field} {query1_comp} {query1_comp_val} select {query1_project_field} as A
    where {query2_comp_field} {query2_comp} {query2_comp_val} select {query2_project_field} as B
    mean A B
(3) where {query1_comp_field} {query1_comp} {query1_comp_val} select {query1_project_field} as A
    where {query2_comp_field} {query2_comp} {query2_comp_val} select {query2_project_field} as B
    diff A B
'''
def parser_len19_type1(table, logicalForm):
    query1_comp_field, query1_comp, query1_comp_val = logicalForm[1], logicalForm[2], logicalForm[3]
    query1_project_field = logicalForm[5]
    queryLogical = ['where', query1_comp_field, query1_comp, query1_comp_val, 'select', query1_project_field]
    result1 = parser_len6(table, queryLogical)

    if result1 == []:
        return NOTFOUND

    query2_comp_field, query2_comp, query2_comp_val = logicalForm[9], logicalForm[10], logicalForm[11]
    query2_project_field = logicalForm[13]
    queryLogical = ['where', query2_comp_field, query2_comp, query2_comp_val, 'select', query2_project_field]
    result2 = parser_len6(table, queryLogical)
    if result2 == []:
        return NOTFOUND

    result = []
    for val1 in result1:
        for val2 in result2:
            if isfloat(val1) == False and isfloat(val2) == False:
                continue
            result.append(str(float(val1) + float(val2)))
    return result

def parser_len19_type2(table, logicalForm):
    queryLogical = [logicalForm[i] for i in range(len(logicalForm))]
    queryLogical[16] = 'sum'
    result1 = parser_len6(table, logicalForm)
    result = []
    for val in result1:
        result.append(str(float(val) / 2.0))
    return result

def parser_len19_type3(table, logicalForm):
    query1_comp_field, query1_comp, query1_comp_val = logicalForm[1], logicalForm[2], logicalForm[3]
    query1_project_field = logicalForm[5]
    queryLogical = ['where', query1_comp_field, query1_comp, query1_comp_val, 'select', query1_project_field]
    result1 = parser_len6(table, queryLogical)
    if result1 == []:
        return NOTFOUND

    query2_comp_field, query2_comp, query2_comp_val = logicalForm[9], logicalForm[10], logicalForm[11]
    query2_project_field = logicalForm[13]
    queryLogical = ['where', query2_comp_field, query2_comp, query2_comp_val, 'select', query2_project_field]
    result2 = parser_len6(table, queryLogical)
    if result2 == []:
        return NOTFOUND

    result = []
    for val1 in result1:
        for val2 in result2:
            if isfloat(val1) == False and isfloat(val2) == False:
                continue
            result.append(str(float(val1) - float(val2)))
    return result

def parser_len19(table, logicalForm):
    if logicalForm[0] != 'where' or logicalForm[4] != 'select' or logicalForm[6] != 'as' or logicalForm[7] != 'A' \
        or logicalForm[8] != 'where' or logicalForm[12] != 'select' or logicalForm[14] != 'as' or logicalForm[15] != 'B'\
        or logicalForm[17] != 'A' or logicalForm[18] != 'B':
        return NOTFOUND
    if logicalForm[16] == 'sum':
        return parser_len19_type1(table, logicalForm)
    elif logicalForm[16] == 'mean':
        return parser_len19_type2(table, logicalForm)
    elif logicalForm[16] == 'diff':
        return parser_len19_type3(table, logicalForm)
    else:
        return NOTFOUND

'''
A function combining all of the above
'''
typeDic = {1: parser_len1, \
    2: parser_len2, \
    3: parser_len3, \
    5: parser_len5, \
    6: parser_len6, \
    7: parser_len7, \
    10: parser_len10, \
    11: parser_len11, \
    13: parser_len13, \
    15: parser_len15, \
    19: parser_len19}


def logicalParser(table, logicalForm):
    form = logicalForm.split()
    if len(form) not in typeDic.keys():
        return NOTFOUND
    return typeDic[len(form)](table, form)

testTable = [
{'Rank':'1', 'Nation':'Venezuela', 'Gold':'5', 'Silver':'2', 'Bronze':'3', 'Total':'10'},
{'Rank':'2', 'Nation':'Colombia', 'Gold':'4', 'Silver':'4', 'Bronze':'9', 'Total':'17'},
{'Rank':'3', 'Nation':'Dominican Republic', 'Gold':'4', 'Silver':'3', 'Bronze':'4', 'Total':'11'},
{'Rank':'4', 'Nation':'Peru', 'Gold':'2', 'Silver':'3', 'Bronze':'3', 'Total':'8'},
{'Rank':'5', 'Nation':'Ecuador', 'Gold':'2', 'Silver':'3', 'Bronze':'3', 'Total':'8'},
{'Rank':'6', 'Nation':'Guatemala', 'Gold':'1', 'Silver':'1', 'Bronze':'1', 'Total':'3'},
{'Rank':'7', 'Nation':'Chile', 'Gold':'0', 'Silver':'3', 'Bronze':'2', 'Total':'5'},
{'Rank':'8', 'Nation':'Panama', 'Gold':'0', 'Silver':'0', 'Bronze':'3', 'Total':'3'},
{'Rank':'9', 'Nation':'Bolivia', 'Gold':'0', 'Silver':'0', 'Bronze':'1', 'Total':'1'},
{'Rank':'9', 'Nation':'Paraguay', 'Gold':'0', 'Silver':'0', 'Bronze':'1', 'Total':'1'},
{'Rank':'Total', 'Nation':'Total', 'Gold':'19', 'Silver':'0', 'Bronze':'35', 'Total':'73'}]

def test_len1_type1(table):
    logicalForm = 'sum'
    print logicalForm + ': ' + str(logicalParser(table, logicalForm))
    return

def test_len2_type1(table):
    logicalForm = 'sum Nation'
    print logicalForm + ': ' + str(logicalParser(table, logicalForm))
    logicalForm = 'sum Gold'
    print logicalForm + ': ' + str(logicalParser(table, logicalForm))
    return

def test_len2_type2(table):
    logicalForm = 'mean Nation'
    print logicalForm + ': ' + str(logicalParser(table, logicalForm))
    logicalForm = 'mean Gold'
    print logicalForm + ': ' + str(logicalParser(table, logicalForm))
    return

def test_len3_type1(table):
    logicalForm = 'argmax Gold Silver'
    print logicalForm + ': ' + str(logicalParser(table, logicalForm))
    logicalForm = 'argmin Total Gold'
    print logicalForm + ': ' + str(logicalParser(table, logicalForm))

def test_len5_type1(table):
    logicalForm = 'where Gold greater 3 sum'
    print logicalForm + ': ' + str(logicalParser(table, logicalForm))
    logicalForm = 'where Total less 2 sum'
    print logicalForm + ': ' + str(logicalParser(table, logicalForm))
    logicalForm = 'I am only a test'
    print logicalForm + ': ' + str(logicalParser(table, logicalForm))

def test_len6_type1(table):
    logicalForm = 'where Gold equal 5 select Nation'
    print logicalForm + ': ' + str(logicalParser(table, logicalForm))
    logicalForm = 'where Total greater 8 select Rank'
    print logicalForm + ': ' + str(logicalParser(table, logicalForm))
    logicalForm = 'where Total greater 8 select Test'
    print logicalForm + ': ' + str(logicalParser(table, logicalForm))
    logicalForm = 'where Total greater Test select Rank'
    print logicalForm + ': ' + str(logicalParser(table, logicalForm))
    return

def test_len6_type2(table):
    logicalForm = 'where Gold equal 5 sum Gold'
    print logicalForm + ': ' + str(logicalParser(table, logicalForm))
    logicalForm = 'where Total greater 8 sum Silver'
    print logicalForm + ': ' + str(logicalParser(table, logicalForm))
    logicalForm = 'where Total greater 8 sum Test'
    print logicalForm + ': ' + str(logicalParser(table, logicalForm))
    logicalForm = 'where Total greater Test sum Rank'
    print logicalForm + ': ' + str(logicalParser(table, logicalForm))
    return

def test_len6_type3(table):
    logicalForm = 'where Gold equal 5 mean Gold'
    print logicalForm + ': ' + str(logicalParser(table, logicalForm))
    logicalForm = 'where Total greater 8 mean Silver'
    print logicalForm + ': ' + str(logicalParser(table, logicalForm))
    logicalForm = 'where Total greater 8 mean Test'
    print logicalForm + ': ' + str(logicalParser(table, logicalForm))
    logicalForm = 'where Total greater Test mean Rank'
    print logicalForm + ': ' + str(logicalParser(table, logicalForm))
    return


def test_len7_type1(table):
    logicalForm = 'where Total less 17 argmax Silver Gold'
    print logicalForm + ': ' + str(logicalParser(table, logicalForm))
    logicalForm = 'where Gold equal 0 argmin Rank Bronze'
    print logicalForm + ': ' + str(logicalParser(table, logicalForm))
    return

def test_len10_type1(table):
    logicalForm = 'where Total greater 10 where Silver less 7 select Rank'
    print logicalForm + ': ' + str(logicalParser(table, logicalForm))
    logicalForm = 'where this is some nonsense and it is really nonsense'
    print logicalForm + ': ' + str(logicalParser(table, logicalForm))
    return

def test_len11_type1(table):
    logicalForm = 'where Total greater 10 where Silver less 7 argmin Rank Gold'
    print logicalForm + ': ' + str(logicalParser(table, logicalForm))
    logicalForm = 'where Silver greater 2 where Total less 10 argmin Rank Bronze'
    print logicalForm + ': ' + str(logicalParser(table, logicalForm))
    return

def test_len13_type1(table):
    logicalForm = 'argmax Gold Silver as A argmin Total Bronze as B diff A B'
    print logicalForm + ': ' + str(logicalParser(table, logicalForm))
    logicalForm = 'argmin Gold Silver as A argmax Total Bronze as B diff A B'
    print logicalForm + ': ' + str(logicalParser(table, logicalForm))
    return

def test_len15_type1(table):
    logicalForm = 'where Gold greater 3 select Silver as A where Silver less A argmin Rank Bronze'
    print logicalForm + ': ' + str(logicalParser(table, logicalForm))
    logicalForm = 'where Gold equal 0 select Silver as A where Silver less A argmin Gold Bronze'
    print logicalForm + ': ' + str(logicalParser(table, logicalForm))
    return


def test_len19_type1(table):
    logicalForm = 'where Gold greater 3 select Silver as A where Total less 10 select Bronze as B sum A B'
    print logicalForm + ': ' + str(logicalParser(table, logicalForm))
    return

def test_len19_type2(table):
    logicalForm = 'where Gold greater 3 select Silver as A where Total less 10 select Bronze as B mean A B'
    print logicalForm + ': ' + str(logicalParser(table, logicalForm))
    return


def test_len19_type3(table):
    logicalForm = 'where Gold greater 3 select Silver as A where Total less 10 select Bronze as B diff A B'
    print logicalForm + ': ' + str(logicalParser(table, logicalForm))
    return


def testFunctions(table):
    print '======================================================'
    print "testing len = 1"
    test_len1_type1(table)
    print '======================================================'
    print "testing len = 2"
    test_len2_type1(table)
    test_len2_type2(table)
    print '======================================================'
    print "testing len = 3"
    test_len3_type1(table)
    print '======================================================'
    print "testing len = 5"
    test_len5_type1(table)
    print '======================================================'
    print "testing len = 6"
    test_len6_type1(table)
    test_len6_type2(table)
    test_len6_type3(table)
    print '======================================================'
    print "testing len = 7"
    test_len7_type1(table)
    print '======================================================'
    print "testing len = 10"
    test_len10_type1(table)
    print '======================================================'
    print "testing len = 11"
    test_len11_type1(table)
    print '======================================================'
    print "testing len = 13"
    test_len13_type1(table)
    print '======================================================'
    print "testing len = 15"
    test_len15_type1(table)
    print '======================================================'
    print "testing len = 19"
    test_len19_type1(table)
    test_len19_type2(table)
    test_len19_type3(table)
    return True


#testFunctions(testTable)

