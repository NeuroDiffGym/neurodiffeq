from lexer import *
import ply.yacc as yacc

start = 'equation'


def p_error(p):
    print(f'Error at token: {p.value}')
    
def p_empty(p):
    'empty :'
    pass


def p_multiplicative_expression(p):
    '''
    multiplicative_expression : multiplicative_expression TIMES coefficient
                              | IDENTIFIER
                              | NUMBER
    '''
    p[0] = ['multiplicative_expression'] + p[1:]
    
# def p_one_expression(p):
#     '''
#     one_expression : NUMBER TIMES IDENTIFIER 
#                    | NUMBER TIMES IDENTIFIER TIMES IDENTIFIER
#                    | IDENTIFIER TIMES IDENTIFIER
#                    | IDENTIFIER TIMES IDENTIFIER TIMES IDENTIFIER
#                    | NUMBER
#                    | IDENTIFIER
#     '''
#     p[0] = ['one_expression'] + p[1:]

def p_expression(p):
    '''
    expression : multiplicative_expression
               | multiplicative_expression PLUS multiplicative_expression
               | multiplicative_expression MINUS multiplicative_expression
    '''
    p[0] = ['expression'] + p[1:]

def p_coefficient(p):
    '''
    coefficient : LEFT_PAREN expression RIGHT_PAREN
                | expression
    '''
    p[0] = ['coefficient'] + p[1:]

def p_term(p):
    '''
    term : coefficient TIMES DIFF LEFT_PAREN expression COMMA IDENTIFIER RIGHT_PAREN
         | DIFF LEFT_PAREN expression COMMA IDENTIFIER RIGHT_PAREN
         | coefficient TIMES DIFF LEFT_PAREN expression COMMA IDENTIFIER COMMA ORDER EQUALS NUMBER RIGHT_PAREN
         | DIFF LEFT_PAREN expression COMMA IDENTIFIER COMMA ORDER EQUALS NUMBER RIGHT_PAREN
         | coefficient
    '''
    p[0] = ['term'] + p[1:]
    
def p_equation(p):
    '''
    equation : term PLUS equation
             | term MINUS equation
             | term
    
    '''
    p[0] = ['equation'] + p[1:]

test_case = '(u*v)*diff(u, t) + x'

lexer.input(test_case)

for t in lexer:
    print(t)

parser = yacc.yacc()
print(parser.parse(test_case))
