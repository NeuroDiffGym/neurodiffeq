import ply.lex as lex

reserved = {
    'diff': 'DIFF',
    'order': 'ORDER',
    'sin': 'SIN',
    'cos': 'COS',
    'tan': 'TAN',
    'sinh': 'SINH',
    'cosh': 'COSH',
    'tanh': 'TANH',
}

greek = {
    'alpha': 'ALPHA',
    'beta': 'BETA',
    'gamma': 'GAMMA',
    'delta': 'DELTA',
    'epsilon': 'EPSILON',
    'zeta': 'ZETA',
    'eta': 'ETA',
    'theta': 'THETA',
    'iota': 'IOTA',
    'kappa': 'KAPPA',
    'lambda': 'LAMBDA',
    'mu': 'MU',
    'nu': 'NU',
    'xi': 'XI',
    'omicron': 'OMICRON',
    'pi': 'PI',
    'rho': 'RHO',
    'sigma': 'SIGMA',
    'tau': 'TAU',
    'upsilon': 'UPSILON',
    'phi': 'PHI',
    'chi': 'CHI',
    'psi': 'PSI',
    'omega': 'OMEGA'
}

tokens = list(reserved.values()) + list(greek.values()) + [
    'POWER',
    'PLUS',
    'MINUS',
    'TIMES',
    'DIVIDE',
    'LEFT_PAREN',
    'RIGHT_PAREN',
    'COMMA',
    'EQUALS',
    'NUMBER',
    'IDENTIFIER'
]

t_POWER = r'\*\*'
t_PLUS    = r'\+'
t_MINUS   = r'-'
t_TIMES   = r'\*'
t_DIVIDE  = r'/'
t_LEFT_PAREN  = r'\('
t_RIGHT_PAREN  = r'\)'
t_COMMA = r','
t_EQUALS = r'='
t_ignore_TABSPACE = r'[ \t]'

def t_NUMBER(t):
    r"\d+(\.\d+)?((e|E)(\+|-)?\d+)?"
    return t

def t_IDENTIFIER(t):
    r"[a-zA-Z_][a-zA-Z0-9_]*"
    t.type = reserved.get(t.value, "IDENTIFIER")
    return t

def t_error(t):
    print(f"Illegal Character found: {t.value[0]} at line {t.lineno}")
    t.lexer.skip(1)
    


test_case = 'diff(u+2,x,order=2) + diff(u,y,order=2) - sin(x)*cos(y)'
lexer = lex.lex()
lexer.input(test_case)
token_list = list(lexer)
print(token_list)
print(len(token_list))

latex_string = ''


def check_pde(token_list):
    independent_var = set()
    i = 0
    while i < len(token_list):
        if token_list[i].type == 'DIFF':
            i += 2
            while token_list[i].type != 'COMMA':
                i += 1
            i += 1
            independent_var.add(token_list[i].value)
        i += 1
    
    return len(independent_var) > 1

def add_token(i, string):
    global token_list
    if token_list[i].type == 'POWER':
        string += '^'
    elif token_list[i].type == 'TIMES':
        string += ''
    elif token_list[i].value in greek.keys():
        string += f'\\{token_list[i].value}'
    else:
        string += token_list[i].value
    return string
        

i = 0

is_pde = check_pde(token_list)

while i < len(token_list):

    
    if token_list[i].type != 'DIFF':
        latex_string = add_token(i, latex_string)
        i += 1
    
    else:
        i += 2
        expression_1 = ''
        while token_list[i].type != 'COMMA':
            expression_1 = add_token(i, expression_1)
            i += 1
        
        i += 1
        expression_2 = ''
        while token_list[i].type != 'COMMA' and token_list[i].type != 'RIGHT_PAREN':
            expression_2 = add_token(i, expression_2)
            i += 1
        
        is_single_exp_l = '(' if len(expression_1) > 1 else ''
        is_single_exp_r = ')' if len(expression_1) > 1 else ''
        
        if token_list[i].type == 'RIGHT_PAREN':
            if is_pde:
                latex_string += '\\frac{\\partial ' + is_single_exp_l + expression_1 + is_single_exp_r + '}{\\partial ' + \
                                expression_2 + '}'
            else:
                latex_string += '\\frac{d' + is_single_exp_l + expression_1 + is_single_exp_r + '}{d ' + \
                                expression_2 + '}'
            i += 1
        else:
            i += 3
            order = int(token_list[i].value)
            if is_pde:
                latex_string += '\\frac{\\partial^' + str(order) + \
                            is_single_exp_l + expression_1 + is_single_exp_r + '}{\\partial ' + expression_2 + '^' + str(order) + '}'
            else:
                latex_string += '\\frac{d^' + str(order) + \
                            is_single_exp_l + expression_1 + is_single_exp_r + '}{d ' + expression_2 + '^' + str(order) + '}'
            i += 2
            
            
print(latex_string)
        
            


