import re

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
    'IDENTIFIER_NUMBER',
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

def t_IDENTIFIER_NUMBER(t):
    r"[a-zA-Z_][a-zA-Z_]*[0-9][0-9]*"
    t.type = reserved.get(t.value, "IDENTIFIER_NUMBER")
    return t

def t_IDENTIFIER(t):
    r"[a-zA-Z_][a-zA-Z0-9_]*"
    t.type = reserved.get(t.value, "IDENTIFIER")
    return t

def t_error(t):
    print(f"Illegal Character found: {t.value[0]} at line {t.lineno}")
    t.lexer.skip(1)
    
def convert_to_latex(token_list):
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
        if token_list[i].type == 'POWER':
            string += '^'
        elif token_list[i].type == 'TIMES':
            # if token_list[i-1].value not in greek.keys() and len(token_list[i-1].value) > 1:
            #     string += ' \\cdot '
            # else:
            #     string += ''
            string += ' \\cdot '
        elif token_list[i].type == 'IDENTIFIER_NUMBER':
            idx = re.search(r"\d", token_list[i].value)
            if idx:
                string += token_list[i].value[:idx.start()] + '_{' + token_list[i].value[idx.start():] + '}'
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
    
    return latex_string


def preprocess(string):
    string = string.replace('torch.', '')
    string = string.replace('np.', '')
    
    lambda_colon = string.find(':')
    if lambda_colon != -1:
        string = string[lambda_colon+1:]
    
    starting_bracket = string.find('[')
    if starting_bracket != -1:
        string = string[starting_bracket:]
    
    string = string.replace('[', '')
    string = string.replace(']', '')
    return string

def split_equations(token_list):
    token_lists = []
    stack = []
    curr_token_list = []
    for i in range(len(token_list)):
        
        if token_list[i].type == 'LEFT_PAREN':
            stack.append(1)
        elif token_list[i].type == 'RIGHT_PAREN':
            stack.pop()
        
        if token_list[i].type == 'COMMA':
            if len(stack) == 0:
                token_lists.append(curr_token_list)
                curr_token_list = []
                continue
        curr_token_list.append(token_list[i])
        
    token_lists.append(curr_token_list)
    return token_lists