# echidna-compiler-
echidna compiler
Skip to content
Chat history
You said:
single file compiled novel compiler
ChatGPT said:

Got it — here’s a single-file, compiled, novel language compiler you can drop in and use. The tiny language (“Quokka”) has fn, let, if/else, while, return, typed params/vars (int, double, bool, string, void), function calls (with simple type-checking + int→double promotion), and a print(expr); statement. It compiles to portable C.
Quick start

python3 quokka.py demo.qk > demo.c
cc demo.c -o demo -lm          # or: clang demo.c -o demo -lm
./demo

Example (demo.qk)

// declare a C math function we want to call
extern fn sin(x: double) -> double;

fn square(x: double) -> double { return x * x; }

fn countdown(n: int) -> void {
  while (n > 0) { print(n); n = n - 1; }
}

fn main() {
  let n: int = 5;
  let a: double = 3.5;
  let ok: bool = true;
  print("Hello, Quokka!");
  print(n);
  print(a);
  print(square(2.0) + sin(0.0));
  if (ok && n >= 5) { print("branch: true"); } else { print("branch: false"); }
  countdown(3);
}

The compiler (quokka.py)

#!/usr/bin/env python3
# Quokka: a tiny one-file compiler from .qk -> C
# Usage: python3 quokka.py input.qk > out.c
# Language:
#   - Types: int, double, bool, string, void
#   - Decls: extern fn f(...)->T;  fn f(...)->T { ... }
#   - Stmts: let name: T (= expr)? ;  name = expr;
#            if (expr) { ... } else { ... }
#            while (expr) { ... }
#            return expr? ;
#            print(expr) ;
#            expr ;   // call only
#   - Exprs: literals (123, 1.23, true, false, "str"), vars, calls
#            unary: - ! ; binary: * / % + - < <= > >= == != && ||
# Notes:
#   - Numeric promotion: int -> double where needed
#   - Assign: double->int is disallowed
#   - main() may omit return; compiler adds `return 0;` in C

import sys, re, textwrap

######################################################################
# Lexing
######################################################################
TT = {
    'EOF':'EOF','ID':'ID','INT':'INT','FLOAT':'FLOAT','STRING':'STRING',
    'LP':'(','RP':')','LB':'{','RB':'}','COMMA':',','SEMI':';','COLON':':',
    'ARROW':'->','ASSIGN':'=',
    'PLUS':'+','MINUS':'-','STAR':'*','SLASH':'/','PERCENT':'%',
    'BANG':'!','LT':'<','LE':'<=','GT':'>','GE':'>=','EQ':'==','NE':'!=',
    'AND':'&&','OR':'||',
}
KEYWORDS = {
    'fn','extern','let','return','if','else','while',
    'true','false','int','double','bool','string','void','print'
}

class Token:
    def __init__(self, kind, val, line, col):
        self.kind, self.val, self.line, self.col = kind, val, line, col
    def __repr__(self): return f'Token({self.kind},{self.val!r},{self.line}:{self.col})'

class Lexer:
    def __init__(self, src):
        self.s, self.i, self.n = src, 0, len(src)
        self.line, self.col = 1, 1
    def _peek(self, k=0):
        j = self.i + k
        return self.s[j] if j < self.n else ''
    def _adv(self):
        ch = self._peek()
        if ch == '\n': self.line += 1; self.col = 1
        else: self.col += 1
        self.i += 1
        return ch
    def _match(self, t):
        if self.s.startswith(t, self.i):
            for _ in t: self._adv()
            return True
        return False
    def tokens(self):
        toks=[]
        while True:
            ch = self._peek()
            if ch == '':
                toks.append(Token(TT['EOF'], '', self.line, self.col)); break
            if ch in ' \t\r\n':
                self._adv(); continue
            if ch == '/' and self._peek(1) == '/':
                while ch and ch != '\n': ch = self._adv()
                continue
            # multi-char operators
            if self._match('->'): toks.append(Token(TT['ARROW'],'->',self.line,self.col)); continue
            if self._match('<='): toks.append(Token(TT['LE'],'<=',self.line,self.col)); continue
            if self._match('>='): toks.append(Token(TT['GE'],'>=',self.line,self.col)); continue
            if self._match('=='): toks.append(Token(TT['EQ'],'==',self.line,self.col)); continue
            if self._match('!='): toks.append(Token(TT['NE'],'!=',self.line,self.col)); continue
            if self._match('&&'): toks.append(Token(TT['AND'],'&&',self.line,self.col)); continue
            if self._match('||'): toks.append(Token(TT['OR'],'||',self.line,self.col)); continue
            # string
            if ch == '"':
                startL, startC = self.line, self.col
                self._adv()
                buf=[]
                while True:
                    c = self._peek()
                    if c == '': raise SyntaxError(f'Unterminated string at {startL}:{startC}')
                    if c == '"': self._adv(); break
                    if c == '\\':
                        self._adv()
                        esc = self._peek()
                        if esc == 'n': buf.append('\n'); self._adv()
                        elif esc == 't': buf.append('\t'); self._adv()
                        elif esc == '"': buf.append('"'); self._adv()
                        elif esc == '\\': buf.append('\\'); self._adv()
                        else: raise SyntaxError(f'Bad escape \\{esc} at {self.line}:{self.col}')
                    else:
                        buf.append(self._adv())
                toks.append(Token(TT['STRING'], ''.join(buf), startL, startC)); continue
            # numbers
            if ch.isdigit():
                startL, startC = self.line, self.col
                num = self._adv()
                isfloat = False
                while self._peek().isdigit(): num += self._adv()
                if self._peek()=='.' and self._peek(1).isdigit():
                    isfloat = True; num += self._adv()
                    while self._peek().isdigit(): num += self._adv()
                if isfloat:
                    toks.append(Token(TT['FLOAT'], float(num), startL, startC))
                else:
                    toks.append(Token(TT['INT'], int(num), startL, startC))
                continue
            # identifiers/keywords
            if ch.isalpha() or ch=='_':
                startL, startC = self.line, self.col
                s = self._adv()
                while self._peek().isalnum() or self._peek()=='_': s += self._adv()
                if s in KEYWORDS:
                    toks.append(Token(s, s, startL, startC))
                else:
                    toks.append(Token(TT['ID'], s, startL, startC))
                continue
            # single-char
            single = {'(':'LP',')':'RP','{':'LB','}':'RB',',':'COMMA',';':'SEMI',':':'COLON',
                      '+':'PLUS','-':'MINUS','*':'STAR','/':'SLASH','%':'PERCENT',
                      '!':'BANG','<':'LT','>':'GT','=':'ASSIGN'}
            if ch in single:
                kind = TT[single[ch]] if single[ch] in TT else single[ch]
                toks.append(Token(kind, self._adv(), self.line, self.col)); continue
            raise SyntaxError(f'Lex error near {self.s[self.i:self.i+20]!r} at {self.line}:{self.col}')
        return toks

######################################################################
# AST + Types
######################################################################
class Type:
    def __init__(self, kind):
        self.kind = kind
    def __repr__(self): return self.kind
    def __eq__(self, o): return isinstance(o, Type) and self.kind==o.kind
T_INT=Type('int'); T_DBL=Type('double'); T_BOOL=Type('bool'); T_STR=Type('string'); T_VOID=Type('void')

def type_from_name(name):
    return {'int':T_INT,'double':T_DBL,'bool':T_BOOL,'string':T_STR,'void':T_VOID}[name]

class Node: pass
class Program(Node):
    def __init__(self, decls): self.decls=decls
class Extern(Node):
    def __init__(self, name, params, ret): self.name=name; self.params=params; self.ret=ret
class Func(Node):
    def __init__(self, name, params, ret, body): self.name=name; self.params=params; self.ret=ret; self.body=body
class Param(Node):
    def __init__(self, name, ty): self.name=name; self.ty=ty
class Block(Node):
    def __init__(self, stmts): self.stmts=stmts
class Let(Node):
    def __init__(self, name, ty, expr): self.name=name; self.ty=ty; self.expr=expr
class Assign(Node):
    def __init__(self, name, expr): self.name=name; self.expr=expr
class If(Node):
    def __init__(self, cond, then, els): self.cond=cond; self.then=then; self.els=els
class While(Node):
    def __init__(self, cond, body): self.cond=cond; self.body=body
class Return(Node):
    def __init__(self, expr): self.expr=expr
class Print(Node):
    def __init__(self, expr): self.expr=expr
class ExprStmt(Node):
    def __init__(self, expr): self.expr=expr
# Exprs
class Var(Node):
    def __init__(self, name): self.name=name; self.ty=None
class Lit(Node):
    def __init__(self, val, ty): self.val=val; self.ty=ty
class Call(Node):
    def __init__(self, name, args): self.name=name; self.args=args; self.ty=None
class Unary(Node):
    def __init__(self, op, rhs): self.op=op; self.rhs=rhs; self.ty=None
class Binary(Node):
    def __init__(self, op, lhs, rhs): self.op=op; self.lhs=lhs; self.rhs=rhs; self.ty=None

######################################################################
# Parser (recursive descent)
######################################################################
class Parser:
    def __init__(self, toks):
        self.toks=toks; self.i=0
    def cur(self): return self.toks[self.i]
    def at(self, kind): return self.cur().kind==kind
    def eat(self, kind, msg=None):
        if not self.at(kind):
            t=self.cur()
            raise SyntaxError(msg or f'Expected {kind}, got {t.kind} at {t.line}:{t.col}')
        tok=self.cur(); self.i+=1; return tok
    def opt(self, kind):
        if self.at(kind): return self.eat(kind)
        return None

    def parse(self):
        decls=[]
        while not self.at('EOF'):
            if self.at('extern'): decls.append(self.parse_extern())
            else: decls.append(self.parse_func())
        return Program(decls)

    def parse_type(self):
        if self.at('int') or self.at('double') or self.at('bool') or self.at('string') or self.at('void'):
            return type_from_name(self.eat(self.cur().kind).val)
        raise SyntaxError(f'Expected type name at {self.cur().line}:{self.cur().col}')

    def parse_params(self):
        params=[]
        if self.at('RP'): return params
        while True:
            name = self.eat(TT['ID']).val
            self.eat(TT['COLON'])
            ty = self.parse_type()
            params.append(Param(name,ty))
            if self.opt(TT['COMMA']): continue
            else: break
        return params

    def parse_extern(self):
        self.eat('extern'); self.eat('fn')
        name = self.eat(TT['ID']).val
        self.eat(TT['LP'])
        params=self.parse_params()
        self.eat(TT['RP'])
        self.eat(TT['ARROW'])
        ret = self.parse_type()
        self.eat(TT['SEMI'])
        return Extern(name,params,ret)

    def parse_func(self):
        self.eat('fn')
        name = self.eat(TT['ID']).val
        self.eat(TT['LP']); params=self.parse_params(); self.eat(TT['RP'])
        ret = T_VOID
        if self.opt(TT['ARROW']): ret=self.parse_type()
        body = self.parse_block()
        return Func(name,params,ret,body)

    def parse_block(self):
        self.eat(TT['LB'])
        stmts=[]
        while not self.at(TT['RB']):
            stmts.append(self.parse_stmt())
        self.eat(TT['RB'])
        return Block(stmts)

    def parse_stmt(self):
        if self.at('let'):
            self.eat('let')
            name = self.eat(TT['ID']).val
            self.eat(TT['COLON'])
            ty = self.parse_type()
            expr=None
            if self.opt(TT['ASSIGN']):
                expr=self.parse_expr()
            self.eat(TT['SEMI'])
            return Let(name,ty,expr)
        if self.at('return'):
            self.eat('return')
            expr=None
            if not self.at(TT['SEMI']):
                expr=self.parse_expr()
            self.eat(TT['SEMI'])
            return Return(expr)
        if self.at('if'):
            self.eat('if'); self.eat(TT['LP']); cond=self.parse_expr(); self.eat(TT['RP'])
            then=self.parse_block()
            els=None
            if self.opt('else'): els=self.parse_block()
            return If(cond,then,els)
        if self.at('while'):
            self.eat('while'); self.eat(TT['LP']); cond=self.parse_expr(); self.eat(TT['RP'])
            body=self.parse_block()
            return While(cond,body)
        if self.at('print'):
            self.eat('print'); self.eat(TT['LP']); e=self.parse_expr(); self.eat(TT['RP']); self.eat(TT['SEMI'])
            return Print(e)
        # assignment or exprstmt
        if self.at(TT['ID']) and self.toks[self.i+1].kind==TT['ASSIGN']:
            name=self.eat(TT['ID']).val; self.eat(TT['ASSIGN']); expr=self.parse_expr(); self.eat(TT['SEMI'])
            return Assign(name,expr)
        # exprstmt (must be call)
        e=self.parse_expr(); self.eat(TT['SEMI'])
        return ExprStmt(e)

    # Expression precedence: OR > AND > EQ > REL > ADD > MUL > UNARY > CALL/PRIMARY
    def parse_expr(self): return self.parse_or()
    def parse_or(self):
        e=self.parse_and()
        while self.opt(TT['OR']): e=Binary('||', e, self.parse_and())
        return e
    def parse_and(self):
        e=self.parse_eq()
        while self.opt(TT['AND']): e=Binary('&&', e, self.parse_eq())
        return e
    def parse_eq(self):
        e=self.parse_rel()
        while True:
            if self.opt(TT['EQ']): e=Binary('==', e, self.parse_rel())
            elif self.opt(TT['NE']): e=Binary('!=', e, self.parse_rel())
            else: break
        return e
    def parse_rel(self):
        e=self.parse_add()
        while True:
            if self.opt(TT['LE']): e=Binary('<=', e, self.parse_add())
            elif self.opt(TT['GE']): e=Binary('>=', e, self.parse_add())
            elif self.opt(TT['LT']): e=Binary('<', e, self.parse_add())
            elif self.opt(TT['GT']): e=Binary('>', e, self.parse_add())
            else: break
        return e
    def parse_add(self):
        e=self.parse_mul()
        while True:
            if self.opt(TT['PLUS']): e=Binary('+', e, self.parse_mul())
            elif self.opt(TT['MINUS']): e=Binary('-', e, self.parse_mul())
            else: break
        return e
    def parse_mul(self):
        e=self.parse_unary()
        while True:
            if self.opt(TT['STAR']): e=Binary('*', e, self.parse_unary())
            elif self.opt(TT['SLASH']): e=Binary('/', e, self.parse_unary())
            elif self.opt(TT['PERCENT']): e=Binary('%', e, self.parse_unary())
            else: break
        return e
    def parse_unary(self):
        if self.opt(TT['MINUS']): return Unary('-', self.parse_unary())
        if self.opt(TT['BANG']): return Unary('!', self.parse_unary())
        return self.parse_call()
    def parse_call(self):
        e=self.parse_primary()
        while self.at(TT['LP']):
            self.eat(TT['LP'])
            args=[]
            if not self.at(TT['RP']):
                while True:
                    args.append(self.parse_expr())
                    if self.opt(TT['COMMA']): continue
                    break
            self.eat(TT['RP'])
            if isinstance(e, Var):
                e = Call(e.name, args)
            else:
                raise SyntaxError(f'Only function calls allowed in this position at {self.cur().line}:{self.cur().col}')
        return e
    def parse_primary(self):
        t=self.cur()
        if self.opt(TT['INT']): return Lit(t.val, T_INT)
        if self.opt(TT['FLOAT']): return Lit(t.val, T_DBL)
        if self.opt('true'): return Lit(1, T_BOOL)
        if self.opt('false'): return Lit(0, T_BOOL)
        if self.opt(TT['STRING']): return Lit(t.val, T_STR)
        if self.opt(TT['ID']): return Var(t.val)
        if self.opt(TT['LP']):
            e=self.parse_expr(); self.eat(TT['RP']); return e
        raise SyntaxError(f'Unexpected token {t.kind} at {t.line}:{t.col}')

######################################################################
# Semantic analysis & type checking
######################################################################
class FuncSig:
    def __init__(self, name, param_types, ret):
        self.name=name; self.param_types=param_types; self.ret=ret

class Context:
    def __init__(self):
        self.funcs={}  # name -> FuncSig
        self.vars_stack=[]  # list of dicts
        self.cur_func=None

    def push_vars(self): self.vars_stack.append({})
    def pop_vars(self): self.vars_stack.pop()
    def declare_var(self, name, ty, where):
        if name in self.vars_stack[-1]:
            raise TypeError(f'Variable {name} already declared {where}')
        self.vars_stack[-1][name]=ty
    def get_var(self, name):
        for d in reversed(self.vars_stack):
            if name in d: return d[name]
        raise TypeError(f'Undeclared variable {name}')
    def declare_func(self, name, sig:FuncSig):
        self.funcs[name]=sig
    def get_func(self, name):
        if name in self.funcs: return self.funcs[name]
        raise TypeError(f'Undeclared function {name}')

def is_numeric(t): return t in (T_INT, T_DBL)
def bin_numeric_type(a,b):
    if a==T_DBL or b==T_DBL: return T_DBL
    return T_INT

def can_assign(dst, src):
    if dst==src: return True
    if dst==T_DBL and src==T_INT: return True
    # forbid double->int implicit, and any other mixes
    return False

def ensure_bool(t, ctx_msg):
    if t!=T_BOOL and not is_numeric(t):
        raise TypeError(f'Condition must be bool/numeric, got {t} in {ctx_msg}')

def analyze_expr(e, ctx:Context):
    if isinstance(e, Lit):
        return e.ty
    if isinstance(e, Var):
        e.ty = ctx.get_var(e.name); return e.ty
    if isinstance(e, Unary):
        rt = analyze_expr(e.rhs, ctx)
        if e.op=='-':
            if not is_numeric(rt): raise TypeError('Unary - expects numeric')
            e.ty = rt; return e.ty
        if e.op=='!':
            if rt not in (T_BOOL, T_INT, T_DBL): raise TypeError('! expects bool/numeric')
            e.ty = T_BOOL; return e.ty
        raise TypeError(f'Unknown unary {e.op}')
    if isinstance(e, Binary):
        lt = analyze_expr(e.lhs, ctx); rt = analyze_expr(e.rhs, ctx)
        op=e.op
        if op in ('*','/','%','+','-'):
            if not (is_numeric(lt) and is_numeric(rt)): raise TypeError(f'Op {op} needs numeric, got {lt},{rt}')
            if op=='%' and (lt!=T_INT or rt!=T_INT): raise TypeError('% requires int')
            e.ty = bin_numeric_type(lt,rt); return e.ty
        if op in ('<','<=','>','>='):
            if not (is_numeric(lt) and is_numeric(rt)): raise TypeError(f'Comparison needs numeric, got {lt},{rt}')
            e.ty = T_BOOL; return e.ty
        if op in ('==','!='):
            if lt!=rt and not (is_numeric(lt) and is_numeric(rt)): raise TypeError('==/!= require same or both numeric')
            e.ty = T_BOOL; return e.ty
        if op in ('&&','||'):
            ensure_bool(lt,'logical op'); ensure_bool(rt,'logical op')
            e.ty = T_BOOL; return e.ty
        raise TypeError(f'Unknown op {op}')
    if isinstance(e, Call):
        sig = ctx.get_func(e.name)
        if len(sig.param_types)!=len(e.args):
            raise TypeError(f'Function {e.name} arity mismatch')
        for i,(at, arg) in enumerate(zip(sig.param_types, e.args)):
            tt = analyze_expr(arg, ctx)
            if at==tt: continue
            if at==T_DBL and tt==T_INT: continue
            raise TypeError(f'Arg {i+1} to {e.name} expects {at}, got {tt}')
        e.ty = sig.ret; return e.ty
    raise TypeError('Unknown expression node')

def analyze_stmt(s, ctx:Context):
    if isinstance(s, Let):
        if s.name in ctx.vars_stack[-1]: raise TypeError(f'Variable {s.name} already declared')
        if s.expr:
            et = analyze_expr(s.expr, ctx)
            if not can_assign(s.ty, et): raise TypeError(f'Cannot assign {et} to {s.ty}')
        ctx.declare_var(s.name, s.ty, 'here')
    elif isinstance(s, Assign):
        ty = ctx.get_var(s.name)
        et = analyze_expr(s.expr, ctx)
        if not can_assign(ty, et): raise TypeError(f'Cannot assign {et} to {ty}')
    elif isinstance(s, If):
        ct = analyze_expr(s.cond, ctx)
        ensure_bool(ct,'if condition')
        ctx.push_vars(); [analyze_stmt(x, ctx) for x in s.then.stmts]; ctx.pop_vars()
        if s.els:
            ctx.push_vars(); [analyze_stmt(x, ctx) for x in s.els.stmts]; ctx.pop_vars()
    elif isinstance(s, While):
        ct = analyze_expr(s.cond, ctx)
        ensure_bool(ct,'while condition')
        ctx.push_vars(); [analyze_stmt(x, ctx) for x in s.body.stmts]; ctx.pop_vars()
    elif isinstance(s, Return):
        if s.expr is None:
            if ctx.cur_func.ret != T_VOID and ctx.cur_func.name!='main':
                raise TypeError('Return without value in non-void function')
        else:
            rt = analyze_expr(s.expr, ctx)
            # allow int->double
            if ctx.cur_func.ret==T_DBL and rt==T_INT: pass
            elif ctx.cur_func.ret != rt:
                raise TypeError(f'Return type {rt} does not match {ctx.cur_func.ret}')
    elif isinstance(s, Print):
        analyze_expr(s.expr, ctx)  # type checked inside expr
    elif isinstance(s, ExprStmt):
        if not isinstance(s.expr, Call):
            raise TypeError('Only function calls allowed as expression statements')
        analyze_expr(s.expr, ctx)
    else:
        raise TypeError('Unknown statement')

def analyze(prog:Program):
    ctx=Context()
    # predeclare externs and functions (signatures)
    for d in prog.decls:
        if isinstance(d, Extern):
            ctx.declare_func(d.name, FuncSig(d.name, [p.ty for p in d.params], d.ret))
        elif isinstance(d, Func):
            ctx.declare_func(d.name, FuncSig(d.name, [p.ty for p in d.params], d.ret))
    # analyze bodies
    for d in prog.decls:
        if isinstance(d, Func):
            ctx.cur_func = ctx.get_func(d.name)
            ctx.push_vars()
            for p in d.params:
                ctx.declare_var(p.name, p.ty, 'param')
            for st in d.body.stmts:
                analyze_stmt(st, ctx)
            ctx.pop_vars()
    # ensure main exists
    if 'main' not in ctx.funcs:
        print('// Warning: no main() defined; producing a library C file.', file=sys.stderr)
    return ctx

######################################################################
# Code generation (to C)
######################################################################
def c_type(t:Type):
    if t==T_INT: return 'long long'
    if t==T_DBL: return 'double'
    if t==T_BOOL: return 'bool'
    if t==T_STR: return 'const char*'
    if t==T_VOID: return 'void'
    raise RuntimeError('unknown type')

class CGen:
    def __init__(self, prog:Program, ctx:Context):
        self.prog=prog; self.ctx=ctx
        self.out=[]
        self.ind=0
        self.has_main = 'main' in ctx.funcs
    def emit(self, s=''):
        self.out.append('  '*self.ind + s)
    def gen(self):
        self.emit('#include <stdio.h>')
        self.emit('#include <math.h>')
        self.emit('#include <stdbool.h>')
        self.emit('#include <string.h>')
        self.emit('')
        self.emit('static inline void __print_int(long long x){ printf("%lld\\n", x); }')
        self.emit('static inline void __print_double(double x){ printf("%.15g\\n", x); }')
        self.emit('static inline void __print_bool(bool x){ printf("%s\\n", x ? "true" : "false"); }')
        self.emit('static inline void __print_string(const char* s){ printf("%s\\n", s); }')
        self.emit('')
        # prototypes
        for d in self.prog.decls:
            if isinstance(d, Extern):
                self.emit(self.sig_to_c(d.name, d.params, d.ret) + ';')
        for d in self.prog.decls:
            if isinstance(d, Func):
                self.emit(self.sig_to_c(d.name, d.params, d.ret) + ';')
        self.emit('')
        # defs
        for d in self.prog.decls:
            if isinstance(d, Func):
                self.gen_func(d)
        return '\n'.join(self.out)

    def sig_to_c(self, name, params, ret):
        # main normalization: if user defined "fn main() {...}" with void default, make int main()
        if name=='main':
            return 'int main(void)'
        ps = ', '.join(f'{c_type(p.ty)} {p.name}' for p in params) or 'void'
        return f'{c_type(ret)} {name}({ps})'

    def gen_func(self, f:Func):
        hdr = self.sig_to_c(f.name, f.params, f.ret)
        self.emit(hdr + ' {'); self.ind+=1
        # local declarations are introduced at first assignment/let; C can handle mixed decls but fine.
        for st in f.body.stmts:
            self.gen_stmt(st)
        if f.name=='main':
            self.emit('return 0;')
        self.ind-=1; self.emit(''); self.emit('}')

    def gen_stmt(self, s):
        if isinstance(s, Let):
            if s.ty==T_STR and s.expr and isinstance(s.expr, Lit) and s.expr.ty==T_STR:
                # string literal can be const char*
                init = f' = "{escape_c_string(s.expr.val)}"' if s.expr else ''
                self.emit(f'{c_type(s.ty)} {s.name}{init};')
            else:
                init = ''
                if s.expr is not None:
                    ex = self.gen_expr(s.expr, want=s.ty)
                    init = f' = {ex}'
                self.emit(f'{c_type(s.ty)} {s.name}{init};')
        elif isinstance(s, Assign):
            self.emit(f'{s.name} = {self.gen_expr(s.expr)};')
        elif isinstance(s, If):
            self.emit(f'if ({self.gen_expr(s.cond, want=T_BOOL)}) {{'); self.ind+=1
            for x in s.then.stmts: self.gen_stmt(x)
            self.ind-=1; self.emit('}')
            if s.els:
                self.emit('else {'); self.ind+=1
                for x in s.els.stmts: self.gen_stmt(x)
                self.ind-=1; self.emit('}')
        elif isinstance(s, While):
            self.emit(f'while ({self.gen_expr(s.cond, want=T_BOOL)}) {{'); self.ind+=1
            for x in s.body.stmts: self.gen_stmt(x)
            self.ind-=1; self.emit('}')
        elif isinstance(s, Return):
            if s.expr is None: self.emit('return;')
            else: self.emit(f'return {self.gen_expr(s.expr)};')
        elif isinstance(s, Print):
            e = s.expr
            # generate based on static type
            ty = self.expr_type(e)
            ce = self.gen_expr(e)
            if ty==T_INT: self.emit(f'__print_int({ce});')
            elif ty==T_DBL: self.emit(f'__print_double({ce});')
            elif ty==T_BOOL: self.emit(f'__print_bool({ce});')
            elif ty==T_STR: self.emit(f'__print_string({ce});')
            else: raise RuntimeError('print unknown type')
        elif isinstance(s, ExprStmt):
            self.emit(self.gen_expr(s.expr) + ';')
        else:
            raise RuntimeError('unknown stmt')

    def expr_type(self, e):
        # Minimal recomputation: use analyzer rules roughly
        if isinstance(e, Lit): return e.ty
        if isinstance(e, Var): return None  # unknown here; rely on gen_expr to read C
        if isinstance(e, Call):
            return self.ctx.get_func(e.name).ret
        if isinstance(e, Unary):
            t = self.expr_type(e.rhs)
            return T_BOOL if e.op=='!' else t
        if isinstance(e, Binary):
            if e.op in ('*','/','+','-'): return T_DBL if (self.expr_type(e.lhs)==T_DBL or self.expr_type(e.rhs)==T_DBL) else T_INT
            if e.op=='%': return T_INT
            if e.op in ('<','<=','>','>=','==','!=','&&','||'): return T_BOOL
        return None

    def gen_expr(self, e, want=None):
        if isinstance(e, Lit):
            if e.ty==T_INT: return str(int(e.val))
            if e.ty==T_DBL:
                s = repr(float(e.val))
                if '.' not in s and 'e' not in s and 'E' not in s: s += '.0'
                return s
            if e.ty==T_BOOL: return 'true' if e.val else 'false'
            if e.ty==T_STR: return f'"{escape_c_string(e.val)}"'
        if isinstance(e, Var):
            return e.name
        if isinstance(e, Unary):
            return f'({e.op}{self.gen_expr(e.rhs)})'
        if isinstance(e, Binary):
            return f'({self.gen_expr(e.lhs)} {e.op} {self.gen_expr(e.rhs)})'
        if isinstance(e, Call):
            args = ', '.join(self.gen_expr(a) for a in e.args)
            return f'{e.name}({args})'
        raise RuntimeError('unknown expr')

def escape_c_string(s):
    return s.replace('\\','\\\\').replace('"','\\"').replace('\n','\\n').replace('\t','\\t')

######################################################################
# Main
######################################################################
def main():
    if len(sys.argv) != 2:
        print('Usage: quokka.py <input.qk>  # outputs C on stdout', file=sys.stderr)
        sys.exit(1)
    src = open(sys.argv[1],'r', encoding='utf-8').read()
    try:
        toks = Lexer(src).tokens()
        ast = Parser(toks).parse()
        ctx = analyze(ast)
        csrc = CGen(ast, ctx).gen()
        sys.stdout.write(csrc)
    except (SyntaxError, TypeError) as e:
        print(f'Compile error: {e}', file=sys.stderr)
        sys.exit(2)

if __name__ == '__main__':
    main()

If you want me to tailor the grammar (e.g., arrays, for, user-defined structs, or compile straight to x86-64 via asm), say the word and I’ll extend this single file.
You said:
3 echc.py demo.ech
Traceback (most recent call last):
  File "/home/ian-martin-ajzenszmidt/Downloads/echc.py", line 401, in <module>
    main()
  File "/home/ian-martin-ajzenszmidt/Downloads/echc.py", line 397, in main
    csrc=compile_source(src)
         ^^^^^^^^^^^^^^^^^^^
  File "/home/ian-martin-ajzenszmidt/Downloads/echc.py", line 388, in compile_source
    toks=lex(src)
         ^^^^^^^^
  File "/home/ian-martin-ajzenszmidt/Downloads/echc.py", line 46, in lex
    raise SyntaxError(f'Lex error near {src[i:i+20]!r}')
SyntaxError: Lex error near '# 36\nprint("a =", a)'
ChatGPT said:

That error means your lexer doesn’t recognise # and chokes as soon as it hits a line-number/comment like:

# 36
print("a =", a)

Add “hash-to-EOL” comments (and optional shebang support) to your lex routine and it’ll compile fine.
Minimal patch (drop-in change)

Right after your existing // ... comment handler in lex, add this:

# allow shell-style / line-number comments:  # anything until newline
if src[i] == '#':
    # (optional) treat a shebang at file start as a comment too
    while i < n and src[i] != '\n':
        i += 1
    continue

If your lexer is class-based with self._peek() / self._adv() (common pattern), use:

if self._peek() == '#':
    # skip until end of line
    while self._peek() and self._peek() != '\n':
        self._adv()
    continue

Put that right below your // comment skipping block.
Then re-run

python3 echc.py demo.ech > demo.c
cc demo.c -o demo -lm
./demo

If you still see a linker error like _start undefined, it means your generated C lacks a main. Ensure your Echidna program defines the entry point your codegen maps to int main(void) (or adjust codegen to emit a tiny main that calls your language’s main() function).
You said:
complete program in one  block
ChatGPT said:

#!/usr/bin/env python3
# echc.py — Echidna: a tiny single-file compiler from .ech -> portable C
# Usage:
#   python3 echc.py demo.ech > demo.c
#   cc demo.c -o demo -lm
# Language features:
#   - Types: int, double, bool, string, void
#   - Decls: extern fn f(...)->T;    fn f(...)->T { ... }
#   - Stmts: let name: T (= expr)? ; name = expr;
#            if (expr) { ... } else { ... }
#            while (expr) { ... }
#            return expr? ;
#            print(expr1, expr2, ...);   # prints space-separated, newline at end
#            expr ;   # call-only expression statement
#   - Expr: literals (123, 1.23, true, false, "str"), vars, calls
#           unary: -  !      binary: * / % + - < <= > >= == != && ||
# Notes:
#   - int auto-promotes to double in expressions/args/returns
#   - No implicit double->int assignment
#   - main() auto-emits `int main(void)` in C and returns 0 if missing return

import sys

######################################################################
# Lexing
######################################################################
TT = {
    'EOF':'EOF','ID':'ID','INT':'INT','FLOAT':'FLOAT','STRING':'STRING',
    'LP':'(','RP':')','LB':'{','RB':'}','COMMA':',','SEMI':';','COLON':':',
    'ARROW':'->','ASSIGN':'=',
    'PLUS':'+','MINUS':'-','STAR':'*','SLASH':'/','PERCENT':'%',
    'BANG':'!','LT':'<','LE':'<=','GT':'>','GE':'>=','EQ':'==','NE':'!=',
    'AND':'&&','OR':'||',
}
KEYWORDS = {'fn','extern','let','return','if','else','while',
            'true','false','int','double','bool','string','void','print'}

class Tok:
    def __init__(self,k,v,l,c): self.k=k; self.v=v; self.l=l; self.c=c
    def __repr__(self): return f'Tok({self.k},{self.v!r},{self.l}:{self.c})'

def lex(src:str):
    i,n,l,c = 0,len(src),1,1
    def peek(k=0): 
        j=i+k
        return src[j] if j<n else ''
    def adv():
        nonlocal i,l,c
        ch = peek()
        if ch=='\n': l+=1; c=1
        else: c+=1
        i+=1
        return ch
    def match(s):
        nonlocal i
        if src.startswith(s,i):
            for _ in s: adv()
            return True
        return False
    toks=[]
    while True:
        ch = peek()
        if ch=='':
            toks.append(Tok('EOF','',l,c)); break
        if ch in ' \t\r\n': adv(); continue
        # comments: // ... and # ... (to end of line), and shebang at file start
        if (ch=='#' and (i==0 or True)) or (ch=='/' and peek(1)=='/'):
            if ch=='/' and peek(1)=='/':
                while peek() and peek()!='\n': adv()
            else:
                while peek() and peek()!='\n': adv()
            continue
        # multi-char ops
        if match('->'): toks.append(Tok('ARROW','->',l,c)); continue
        if match('<='): toks.append(Tok('LE','<=',l,c)); continue
        if match('>='): toks.append(Tok('GE','>=',l,c)); continue
        if match('=='): toks.append(Tok('EQ','==',l,c)); continue
        if match('!='): toks.append(Tok('NE','!=',l,c)); continue
        if match('&&'): toks.append(Tok('AND','&&',l,c)); continue
        if match('||'): toks.append(Tok('OR','||',l,c)); continue
        # string
        if ch=='"':
            sl,sc = l,c
            adv()  # "
            buf=[]
            while True:
                p=peek()
                if p=='': raise SyntaxError(f'Unterminated string at {sl}:{sc}')
                if p=='"': adv(); break
                if p=='\\':
                    adv()
                    q=peek()
                    if   q=='n': buf.append('\n'); adv()
                    elif q=='t': buf.append('\t'); adv()
                    elif q=='"': buf.append('"');  adv()
                    elif q=='\\': buf.append('\\'); adv()
                    else: raise SyntaxError(f'Bad escape \\{q} at {l}:{c}')
                else:
                    buf.append(adv())
            toks.append(Tok('STRING',''.join(buf),sl,sc)); continue
        # number
        if ch.isdigit():
            sl,sc = l,c
            s=adv()
            isf=False
            while peek().isdigit(): s+=adv()
            if peek()=='.' and peek(1).isdigit():
                isf=True; s+=adv()
                while peek().isdigit(): s+=adv()
            toks.append(Tok('FLOAT' if isf else 'INT', float(s) if isf else int(s), sl, sc))
            continue
        # id/kw
        if ch.isalpha() or ch=='_':
            sl,sc=l,c
            s=adv()
            while peek().isalnum() or peek()=='_': s+=adv()
            if s in KEYWORDS: toks.append(Tok(s,s,sl,sc))
            else: toks.append(Tok('ID',s,sl,sc))
            continue
        # single-char
        single = {'(':'LP',')':'RP','{':'LB','}':'RB',',':'COMMA',';':'SEMI',':':'COLON',
                  '+':'PLUS','-':'MINUS','*':'STAR','/':'SLASH','%':'PERCENT',
                  '!':'BANG','<':'LT','>':'GT','=':'ASSIGN'}
        if ch in single:
            k = single[ch]; adv(); toks.append(Tok(k, ch, l, c)); continue
        raise SyntaxError(f'Lex error near {src[i:i+20]!r]}')
    return toks

######################################################################
# AST & Types
######################################################################
class Type:
    def __init__(self,k): self.k=k
    def __eq__(self,o): return isinstance(o,Type) and self.k==o.k
    def __repr__(self): return self.k
T_INT=Type('int'); T_DBL=Type('double'); T_BOOL=Type('bool'); T_STR=Type('string'); T_VOID=Type('void')
def type_from_name(n): return {'int':T_INT,'double':T_DBL,'bool':T_BOOL,'string':T_STR,'void':T_VOID}[n]

class Node: pass
class Program(Node):  def __init__(self,decls): self.decls=decls
class Extern(Node):   def __init__(self,name,params,ret): self.name=name; self.params=params; self.ret=ret
class Func(Node):     def __init__(self,name,params,ret,body): self.name=name; self.params=params; self.ret=ret; self.body=body
class Param(Node):    def __init__(self,name,ty): self.name=name; self.ty=ty
class Block(Node):    def __init__(self,stmts): self.stmts=stmts
class Let(Node):      def __init__(self,name,ty,expr): self.name=name; self.ty=ty; self.expr=expr
class Assign(Node):   def __init__(self,name,expr): self.name=name; self.expr=expr
class If(Node):       def __init__(self,cond,then,els): self.cond=cond; self.then=then; self.els=els
class While(Node):    def __init__(self,cond,body): self.cond=cond; self.body=body
class Return(Node):   def __init__(self,expr): self.expr=expr
class Print(Node):    def __init__(self,args): self.args=args
class ExprStmt(Node): def __init__(self,expr): self.expr=expr
# Exprs
class Var(Node):      def __init__(self,name): self.name=name; self.ty=None
class Lit(Node):      def __init__(self,val,ty): self.val=val; self.ty=ty
class Call(Node):     def __init__(self,name,args): self.name=name; self.args=args; self.ty=None
class Unary(Node):    def __init__(self,op,rhs): self.op=op; self.rhs=rhs; self.ty=None
class Binary(Node):   def __init__(self,op,lhs,rhs): self.op=op; self.lhs=lhs; self.rhs=rhs; self.ty=None

######################################################################
# Parser
######################################################################
class Parser:
    def __init__(self,toks): self.toks=toks; self.i=0
    def cur(self): return self.toks[self.i]
    def at(self,k): return self.cur().k==k
    def eat(self,k,msg=None):
        if not self.at(k): t=self.cur(); raise SyntaxError(msg or f'Expected {k}, got {t.k} at {t.l}:{t.c}')
        tok=self.cur(); self.i+=1; return tok
    def opt(self,k): 
        if self.at(k): return self.eat(k)
        return None

    def parse(self):
        ds=[]
        while not self.at('EOF'):
            if self.at('extern'): ds.append(self.parse_extern())
            else: ds.append(self.parse_func())
        return Program(ds)

    def parse_type(self):
        if self.at('int') or self.at('double') or self.at('bool') or self.at('string') or self.at('void'):
            return type_from_name(self.eat(self.cur().k).v)
        raise SyntaxError(f'Expected type at {self.cur().l}:{self.cur().c}')

    def parse_params(self):
        ps=[]
        if self.at('RP'): return ps
        while True:
            name = self.eat('ID').v
            self.eat('COLON'); ty=self.parse_type()
            ps.append(Param(name,ty))
            if self.opt('COMMA'): continue
            break
        return ps

    def parse_extern(self):
        self.eat('extern'); self.eat('fn')
        name=self.eat('ID').v
        self.eat('LP'); ps=self.parse_params(); self.eat('RP')
        self.eat('ARROW'); ret=self.parse_type()
        self.eat('SEMI')
        return Extern(name,ps,ret)

    def parse_func(self):
        self.eat('fn'); name=self.eat('ID').v
        self.eat('LP'); ps=self.parse_params(); self.eat('RP')
        ret=T_VOID
        if self.opt('ARROW'): ret=self.parse_type()
        body=self.parse_block()
        return Func(name,ps,ret,body)

    def parse_block(self):
        self.eat('LB'); ss=[]
        while not self.at('RB'): ss.append(self.parse_stmt())
        self.eat('RB'); return Block(ss)

    def parse_stmt(self):
        if self.at('let'):
            self.eat('let'); name=self.eat('ID').v
            self.eat('COLON'); ty=self.parse_type()
            expr=None
            if self.opt('ASSIGN'): expr=self.parse_expr()
            self.eat('SEMI'); return Let(name,ty,expr)
        if self.at('return'):
            self.eat('return')
            e=None
            if not self.at('SEMI'): e=self.parse_expr()
            self.eat('SEMI'); return Return(e)
        if self.at('if'):
            self.eat('if'); self.eat('LP'); cond=self.parse_expr(); self.eat('RP')
            then=self.parse_block(); els=None
            if self.opt('else'): els=self.parse_block()
            return If(cond,then,els)
        if self.at('while'):
            self.eat('while'); self.eat('LP'); cond=self.parse_expr(); self.eat('RP')
            body=self.parse_block(); return While(cond,body)
        if self.at('print'):
            self.eat('print'); self.eat('LP')
            args=[]
            if not self.at('RP'):
                while True:
                    args.append(self.parse_expr())
                    if self.opt('COMMA'): continue
                    break
            self.eat('RP'); self.eat('SEMI')
            return Print(args)
        # assign
        if self.at('ID') and self.toks[self.i+1].k=='ASSIGN':
            name=self.eat('ID').v; self.eat('ASSIGN'); e=self.parse_expr(); self.eat('SEMI')
            return Assign(name,e)
        # exprstmt (call only)
        e=self.parse_expr(); self.eat('SEMI'); return ExprStmt(e)

    # precedence
    def parse_expr(self): return self.parse_or()
    def parse_or(self):
        e=self.parse_and()
        while self.opt('OR'): e=Binary('||',e,self.parse_and())
        return e
    def parse_and(self):
        e=self.parse_eq()
        while self.opt('AND'): e=Binary('&&',e,self.parse_eq())
        return e
    def parse_eq(self):
        e=self.parse_rel()
        while True:
            if self.opt('EQ'): e=Binary('==',e,self.parse_rel())
            elif self.opt('NE'): e=Binary('!=',e,self.parse_rel())
            else: break
        return e
    def parse_rel(self):
        e=self.parse_add()
        while True:
            if self.opt('LE'): e=Binary('<=',e,self.parse_add())
            elif self.opt('GE'): e=Binary('>=',e,self.parse_add())
            elif self.opt('LT'): e=Binary('<',e,self.parse_add())
            elif self.opt('GT'): e=Binary('>',e,self.parse_add())
            else: break
        return e
    def parse_add(self):
        e=self.parse_mul()
        while True:
            if self.opt('PLUS'): e=Binary('+',e,self.parse_mul())
            elif self.opt('MINUS'): e=Binary('-',e,self.parse_mul())
            else: break
        return e
    def parse_mul(self):
        e=self.parse_un()
        while True:
            if self.opt('STAR'): e=Binary('*',e,self.parse_un())
            elif self.opt('SLASH'): e=Binary('/',e,self.parse_un())
            elif self.opt('PERCENT'): e=Binary('%',e,self.parse_un())
            else: break
        return e
    def parse_un(self):
        if self.opt('MINUS'): return Unary('-', self.parse_un())
        if self.opt('BANG'):  return Unary('!', self.parse_un())
        return self.parse_call()
    def parse_call(self):
        e=self.parse_primary()
        while self.at('LP'):
            self.eat('LP')
            args=[]
            if not self.at('RP'):
                while True:
                    args.append(self.parse_expr())
                    if self.opt('COMMA'): continue
                    break
            self.eat('RP')
            if isinstance(e,Var): e=Call(e.name,args)
            else: raise SyntaxError(f'Call target must be a name at {self.cur().l}:{self.cur().c}')
        return e
    def parse_primary(self):
        t=self.cur()
        if self.opt('INT'): return Lit(t.v,T_INT)
        if self.opt('FLOAT'): return Lit(t.v,T_DBL)
        if self.opt('STRING'): return Lit(t.v,T_STR)
        if self.opt('true'): return Lit(1,T_BOOL)
        if self.opt('false'):return Lit(0,T_BOOL)
        if self.opt('ID'): return Var(t.v)
        if self.opt('LP'):
            e=self.parse_expr(); self.eat('RP'); return e
        raise SyntaxError(f'Unexpected token {t.k} at {t.l}:{t.c}')

######################################################################
# Semantics
######################################################################
class FuncSig:
    def __init__(self,name,ps,ret): self.name=name; self.param_types=ps; self.ret=ret
class Ctx:
    def __init__(self):
        self.funcs={}
        self.vars=[]
        self.cur=None
    def push(self): self.vars.append({})
    def pop(self): self.vars.pop()
    def decl_var(self,n,t):
        if n in self.vars[-1]: raise TypeError(f'Variable {n} already declared')
        self.vars[-1][n]=t
    def get_var(self,n):
        for d in reversed(self.vars):
            if n in d: return d[n]
        raise TypeError(f'Undeclared variable {n}')
    def decl_fn(self,name,sig): self.funcs[name]=sig
    def get_fn(self,name):
        if name in self.funcs: return self.funcs[name]
        raise TypeError(f'Undeclared function {name}')

def is_num(t): return t in (T_INT,T_DBL)
def bin_num_type(a,b): return T_DBL if (a==T_DBL or b==T_DBL) else T_INT
def can_assign(dst,src): return dst==src or (dst==T_DBL and src==T_INT)

def ensure_boolish(t,ctxmsg):
    if t not in (T_BOOL,T_INT,T_DBL): raise TypeError(f'Condition must be bool/numeric in {ctxmsg}, got {t}')

def analyze_expr(e,ctx:Ctx):
    if isinstance(e,Lit): return e.ty
    if isinstance(e,Var):
        e.ty=ctx.get_var(e.name); return e.ty
    if isinstance(e,Unary):
        rt=analyze_expr(e.rhs,ctx)
        if e.op=='-':
            if not is_num(rt): raise TypeError('Unary - expects numeric')
            e.ty=rt; return e.ty
        if e.op=='!':
            if rt not in (T_BOOL,T_INT,T_DBL): raise TypeError('! expects bool/numeric')
            e.ty=T_BOOL; return e.ty
    if isinstance(e,Binary):
        lt=analyze_expr(e.lhs,ctx); rt=analyze_expr(e.rhs,ctx); op=e.op
        if op in ('*','/','+','-'):
            if not (is_num(lt) and is_num(rt)): raise TypeError(f'Op {op} needs numeric')
            e.ty=bin_num_type(lt,rt); return e.ty
        if op=='%':
            if lt!=T_INT or rt!=T_INT: raise TypeError('% requires int')
            e.ty=T_INT; return e.ty
        if op in ('<','<=','>','>=','==','!='):
            if op in ('==','!='):
                if lt!=rt and not (is_num(lt) and is_num(rt)): raise TypeError('==/!= need same or both numeric')
            else:
                if not (is_num(lt) and is_num(rt)): raise TypeError('Comparison needs numeric')
            e.ty=T_BOOL; return e.ty
        if op in ('&&','||'):
            ensure_boolish(lt,'logical'); ensure_boolish(rt,'logical'); e.ty=T_BOOL; return e.ty
    if isinstance(e,Call):
        sig=ctx.get_fn(e.name)
        if len(sig.param_types)!=len(e.args): raise TypeError(f'Arity mismatch for {e.name}')
        for i,(pt,a) in enumerate(zip(sig.param_types,e.args),1):
            at=analyze_expr(a,ctx)
            if pt==at: continue
            if pt==T_DBL and at==T_INT: continue
            raise TypeError(f'Arg {i} to {e.name} expects {pt}, got {at}')
        e.ty=sig.ret; return e.ty
    raise TypeError('Unknown expression')

def analyze_stmt(s,ctx:Ctx):
    if isinstance(s,Let):
        if s.expr:
            et=analyze_expr(s.expr,ctx)
            if not can_assign(s.ty,et): raise TypeError(f'Cannot assign {et} to {s.ty}')
        ctx.decl_var(s.name,s.ty)
    elif isinstance(s,Assign):
        vt=ctx.get_var(s.name); et=analyze_expr(s.expr,ctx)
        if not can_assign(vt,et): raise TypeError(f'Cannot assign {et} to {vt}')
    elif isinstance(s,If):
        ct=analyze_expr(s.cond,ctx); ensure_boolish(ct,'if')
        ctx.push(); [analyze_stmt(x,ctx) for x in s.then.stmts]; ctx.pop()
        if s.els: ctx.push(); [analyze_stmt(x,ctx) for x in s.els.stmts]; ctx.pop()
    elif isinstance(s,While):
        ct=analyze_expr(s.cond,ctx); ensure_boolish(ct,'while')
        ctx.push(); [analyze_stmt(x,ctx) for x in s.body.stmts]; ctx.pop()
    elif isinstance(s,Return):
        if s.expr is None:
            if ctx.cur.ret!=T_VOID and ctx.cur.name!='main': raise TypeError('Return without value')
        else:
            rt=analyze_expr(s.expr,ctx)
            if ctx.cur.ret==T_DBL and rt==T_INT: pass
            elif ctx.cur.ret!=rt: raise TypeError(f'Return type {rt} != {ctx.cur.ret}')
    elif isinstance(s,Print):
        for a in s.args: analyze_expr(a,ctx)  # ensure args are well-typed
    elif isinstance(s,ExprStmt):
        if not isinstance(s.expr,Call): raise TypeError('Only function calls allowed as expression statements')
        analyze_expr(s.expr,ctx)
    else: raise TypeError('Unknown statement')

def analyze(prog:Program):
    ctx=Ctx()
    # declare all functions first
    for d in prog.decls:
        if isinstance(d,Extern) or isinstance(d,Func):
            ctx.decl_fn(d.name, FuncSig(d.name, [p.ty for p in d.params], d.ret if isinstance(d,Extern) else d.ret))
    # analyze bodies
    for d in prog.decls:
        if isinstance(d,Func):
            ctx.cur = ctx.get_fn(d.name)
            ctx.push()
            for p in d.params: ctx.decl_var(p.name,p.ty)
            for st in d.body.stmts: analyze_stmt(st,ctx)
            ctx.pop()
    if 'main' not in ctx.funcs:
        print('// Warning: no main() defined; producing a library C file.', file=sys.stderr)
    return ctx

######################################################################
# Codegen (C)
######################################################################
def c_type(t:Type):
    if t==T_INT: return 'long long'
    if t==T_DBL: return 'double'
    if t==T_BOOL: return 'bool'
    if t==T_STR: return 'const char*'
    if t==T_VOID: return 'void'
    raise RuntimeError('unknown type')

def c_escape(s:str):
    return s.replace('\\','\\\\').replace('"','\\"').replace('\n','\\n').replace('\t','\\t')

class CGen:
    def __init__(self,prog:Program,ctx:Ctx):
        self.p=prog; self.ctx=ctx; self.out=[]; self.ind=0
    def emit(self,s=''): self.out.append('  '*self.ind + s)
    def gen(self):
        self.emit('#include <stdio.h>')
        self.emit('#include <math.h>')
        self.emit('#include <stdbool.h>')
        self.emit('#include <string.h>')
        self.emit('')
        # print helpers (no EOL)
        self.emit('static inline void __p_int(long long x){ printf("%lld", x); }')
        self.emit('static inline void __p_double(double x){ printf("%.15g", x); }')
        self.emit('static inline void __p_bool(bool x){ printf("%s", x ? "true" : "false"); }')
        self.emit('static inline void __p_string(const char* s){ printf("%s", s); }')
        self.emit('')
        # prototypes
        for d in self.p.decls:
            if isinstance(d,Extern): self.emit(self.sig_to_c(d.name,d.params,d.ret) + ';')
        for d in self.p.decls:
            if isinstance(d,Func): self.emit(self.sig_to_c(d.name,d.params,d.ret) + ';')
        self.emit('')
        # defs
        for d in self.p.decls:
            if isinstance(d,Func): self.gen_func(d)
        return '\n'.join(self.out)

    def sig_to_c(self,name,ps,ret):
        if name=='main': return 'int main(void)'
        params = ', '.join(f'{c_type(p.ty)} {p.name}' for p in ps) or 'void'
        return f'{c_type(ret)} {name}({params})'

    def gen_func(self,f:Func):
        self.emit(self.sig_to_c(f.name,f.params,f.ret) + ' {'); self.ind+=1
        for st in f.body.stmts: self.gen_stmt(st)
        if f.name=='main': self.emit('return 0;')
        self.ind-=1; self.emit('}'); self.emit('')

    def gen_stmt(self,s):
        if isinstance(s,Let):
            init=''
            if s.expr is not None: init = f' = {self.gen_expr(s.expr, want=s.ty)}'
            self.emit(f'{c_type(s.ty)} {s.name}{init};')
        elif isinstance(s,Assign):
            self.emit(f'{s.name} = {self.gen_expr(s.expr)};')
        elif isinstance(s,If):
            self.emit(f'if ({self.gen_expr(s.cond, want=T_BOOL)}) {{'); self.ind+=1
            for x in s.then.stmts: self.gen_stmt(x)
            self.ind-=1; self.emit('}')
            if s.els:
                self.emit('else {'); self.ind+=1
                for x in s.els.stmts: self.gen_stmt(x)
                self.ind-=1; self.emit('}')
        elif isinstance(s,While):
            self.emit(f'while ({self.gen_expr(s.cond, want=T_BOOL)}) {{'); self.ind+=1
            for x in s.body.stmts: self.gen_stmt(x)
            self.ind-=1; self.emit('}')
        elif isinstance(s,Return):
            if s.expr is None: self.emit('return;')
            else: self.emit(f'return {self.gen_expr(s.expr)};')
        elif isinstance(s,Print):
            # print args space-separated, then newline
            first=True
            for a in s.args:
                if not first: self.emit('printf(" ");')
                first=False
                ty=self.expr_static_type(a)
                ce=self.gen_expr(a)
                if ty==T_INT: self.emit(f'__p_int({ce});')
                elif ty==T_DBL: self.emit(f'__p_double({ce});')
                elif ty==T_BOOL: self.emit(f'__p_bool({ce});')
                elif ty==T_STR: self.emit(f'__p_string({ce});')
                else: self.emit(f'/* unknown print type */')
            self.emit('printf("\\n");')
        elif isinstance(s,ExprStmt):
            self.emit(self.gen_expr(s.expr) + ';')
        else:
            raise RuntimeError('unknown stmt')

    def expr_static_type(self,e):
        if isinstance(e,Lit): return e.ty
        if isinstance(e,Call): return self.ctx.get_fn(e.name).ret
        if isinstance(e,Unary):
            t=self.expr_static_type(e.rhs)
            return T_BOOL if e.op=='!' else t
        if isinstance(e,Binary):
            if e.op in ('*','/','+','-'): return T_DBL
            if e.op=='%': return T_INT
            return T_BOOL
        return None

    def gen_expr(self,e,want=None):
        if isinstance(e,Lit):
            if e.ty==T_INT: return str(int(e.val))
            if e.ty==T_DBL:
                s=repr(float(e.val))
                if '.' not in s and 'e' not in s and 'E' not in s: s+='.0'
                return s
            if e.ty==T_BOOL: return 'true' if e.val else 'false'
            if e.ty==T_STR: return f'"{c_escape(e.val)}"'
        if isinstance(e,Var): return e.name
        if isinstance(e,Unary): return f'({e.op}{self.gen_expr(e.rhs)})'
        if isinstance(e,Binary): return f'({self.gen_expr(e.lhs)} {e.op} {self.gen_expr(e.rhs)})'
        if isinstance(e,Call):
            args=', '.join(self.gen_expr(a) for a in e.args)
            return f'{e.name}({args})'
        raise RuntimeError('unknown expr')

######################################################################
# Driver
######################################################################
def compile_source(src:str)->str:
    toks=lex(src)
    ast=Parser(toks).parse()
    ctx=analyze(ast)
    csrc=CGen(ast,ctx).gen()
    return csrc

def main():
    if len(sys.argv)!=2:
        print('Usage: echc.py <input.ech>  # outputs C on stdout', file=sys.stderr)
        sys.exit(1)
    src=open(sys.argv[1],'r',encoding='utf-8').read()
    try:
        sys.stdout.write(compile_source(src))
    except (SyntaxError,TypeError) as e:
        print(f'Compile error: {e}', file=sys.stderr)
        sys.exit(2)

if __name__=='__main__':
    main()

You said:
(base) ian-martin-ajzenszmidt@ian-martin-ajzenszmidt-CFSZ5-3:~/Downloads$ python3 echc.py hello.ech
Traceback (most recent call last):
  File "/home/ian-martin-ajzenszmidt/Downloads/echc.py", line 401, in <module>
    main()
  File "/home/ian-martin-ajzenszmidt/Downloads/echc.py", line 397, in main
    csrc=compile_source(src)
         ^^^^^^^^^^^^^^^^^^^
  File "/home/ian-martin-ajzenszmidt/Downloads/echc.py", line 388, in compile_source
    toks=lex(src)
         ^^^^^^^^
  File "/home/ian-martin-ajzenszmidt/Downloads/echc.py", line 46, in lex
    raise SyntaxError(f'Lex error near {src[i:i+20]!r}')
SyntaxError: Lex error near '"Hello, world!");\n'
ChatGPT said:

You’re still running the old echc.py—that lexer doesn’t handle string literals, so it dies at "Hello, world!". Replace your file with the complete compiler below (it supports "#"/"//" comments and double-quoted strings), then try again.

hello.ech

fn main() {
  print("Hello, world!");
}

echc.py (drop-in, single file)

#!/usr/bin/env python3
# echc.py — Echidna: tiny single-file compiler from .ech -> portable C
# Usage:
#   python3 echc.py hello.ech > hello.c
#   cc hello.c -o hello -lm
# Language:
#   Types: int, double, bool, string, void
#   Decls: extern fn f(...)->T;  fn f(...)->T { ... }
#   Stmts: let name: T (= expr)? ;  name = expr;
#          if (expr) { ... } else { ... }
#          while (expr) { ... }
#          return expr? ;
#          print(expr[, expr...]);   # space-separated, newline
#          expr ;   # call-only stmt
#   Expr: literals (123, 1.23, true, false, "str"), vars, calls
#         unary: - ! ;  binary: * / % + - < <= > >= == != && ||

import sys

# ============================== Lexing ==============================
TT = {
  'EOF':'EOF','ID':'ID','INT':'INT','FLOAT':'FLOAT','STRING':'STRING',
  'LP':'(','RP':')','LB':'{','RB':'}','COMMA':',','SEMI':';','COLON':':',
  'ARROW':'->','ASSIGN':'=','PLUS':'+','MINUS':'-','STAR':'*','SLASH':'/',
  'PERCENT':'%','BANG':'!','LT':'<','LE':'<=','GT':'>','GE':'>=',
  'EQ':'==','NE':'!=','AND':'&&','OR':'||',
}
KEYWORDS = {'fn','extern','let','return','if','else','while',
            'true','false','int','double','bool','string','void','print'}

class Tok:
    def __init__(self,k,v,l,c): self.k=k; self.v=v; self.l=l; self.c=c

def lex(src:str):
    i,n,l,c = 0,len(src),1,1
    def peek(k=0): j=i+k; return src[j] if j<n else ''
    def adv():
        nonlocal i,l,c
        ch = peek()
        if ch=='\n': l+=1; c=1
        else: c+=1
        i+=1
        return ch
    def match(s):
        nonlocal i
        if src.startswith(s,i):
            for _ in s: adv()
            return True
        return False
    toks=[]
    while True:
        ch = peek()
        if ch=='':
            toks.append(Tok('EOF','',l,c)); break
        if ch in ' \t\r\n': adv(); continue
        # comments: //... and #...
        if (ch=='/' and peek(1)=='/') or ch=='#':
            while peek() and peek()!='\n': adv()
            continue
        # multi-char ops
        if   match('->'): toks.append(Tok('ARROW','->',l,c)); continue
        if   match('<='): toks.append(Tok('LE','<=',l,c)); continue
        if   match('>='): toks.append(Tok('GE','>=',l,c)); continue
        if   match('=='): toks.append(Tok('EQ','==',l,c)); continue
        if   match('!='): toks.append(Tok('NE','!=',l,c)); continue
        if   match('&&'): toks.append(Tok('AND','&&',l,c)); continue
        if   match('||'): toks.append(Tok('OR','||',l,c)); continue
        # string
        if ch=='"':
            sl,sc=l,c; adv()
            buf=[]
            while True:
                p=peek()
                if p=='': raise SyntaxError(f'Unterminated string at {sl}:{sc}')
                if p=='"': adv(); break
                if p=='\\':
                    adv(); q=peek()
                    if   q=='n': buf.append('\n'); adv()
                    elif q=='t': buf.append('\t'); adv()
                    elif q=='"': buf.append('"');  adv()
                    elif q=='\\': buf.append('\\'); adv()
                    else: raise SyntaxError(f'Bad escape \\{q} at {l}:{c}')
                else:
                    buf.append(adv())
            toks.append(Tok('STRING',''.join(buf),sl,sc)); continue
        # number
        if ch.isdigit():
            sl,sc=l,c; s=adv(); isf=False
            while peek().isdigit(): s+=adv()
            if peek()=='.' and peek(1).isdigit():
                isf=True; s+=adv()
                while peek().isdigit(): s+=adv()
            toks.append(Tok('FLOAT' if isf else 'INT', float(s) if isf else int(s), sl, sc)); continue
        # id / keyword
        if ch.isalpha() or ch=='_':
            sl,sc=l,c; s=adv()
            while peek().isalnum() or peek()=='_': s+=adv()
            toks.append(Tok(s if s in KEYWORDS else 'ID', s if s in KEYWORDS else s, sl, sc)); continue
        # single-char
        single = {'(':'LP',')':'RP','{':'LB','}':'RB',',':'COMMA',';':'SEMI',':':'COLON',
                  '+':'PLUS','-':'MINUS','*':'STAR','/':'SLASH','%':'PERCENT',
                  '!':'BANG','<':'LT','>':'GT','=':'ASSIGN'}
        if ch in single:
            k = single[ch]; adv(); toks.append(Tok(k,ch,l,c)); continue
        raise SyntaxError(f'Lex error near {src[i:i+20]!r}')
    return toks

# ========================== AST & Types ============================
class Type:
    def __init__(self,k): self.k=k
    def __eq__(self,o): return isinstance(o,Type) and self.k==o.k
    def __repr__(self): return self.k
T_INT=Type('int'); T_DBL=Type('double'); T_BOOL=Type('bool'); T_STR=Type('string'); T_VOID=Type('void')
def type_from_name(n): return {'int':T_INT,'double':T_DBL,'bool':T_BOOL,'string':T_STR,'void':T_VOID}[n]

class Node: pass
class Program(Node):  def __init__(self,decls): self.decls=decls
class Extern(Node):   def __init__(self,name,params,ret): self.name=name; self.params=params; self.ret=ret
class Func(Node):     def __init__(self,name,params,ret,body): self.name=name; self.params=params; self.ret=ret; self.body=body
class Param(Node):    def __init__(self,name,ty): self.name=name; self.ty=ty
class Block(Node):    def __init__(self,stmts): self.stmts=stmts
class Let(Node):      def __init__(self,name,ty,expr): self.name=name; self.ty=ty; self.expr=expr
class Assign(Node):   def __init__(self,name,expr): self.name=name; self.expr=expr
class If(Node):       def __init__(self,cond,then,els): self.cond=cond; self.then=then; self.els=els
class While(Node):    def __init__(self,cond,body): self.cond=cond; self.body=body
class Return(Node):   def __init__(self,expr): self.expr=expr
class Print(Node):    def __init__(self,args): self.args=args
class ExprStmt(Node): def __init__(self,expr): self.expr=expr
# Exprs
class Var(Node):      def __init__(self,name): self.name=name; self.ty=None
class Lit(Node):      def __init__(self,val,ty): self.val=val; self.ty=ty
class Call(Node):     def __init__(self,name,args): self.name=name; self.args=args; self.ty=None
class Unary(Node):    def __init__(self,op,rhs): self.op=op; self.rhs=rhs; self.ty=None
class Binary(Node):   def __init__(self,op,lhs,rhs): self.op=op; self.lhs=lhs; self.rhs=rhs; self.ty=None

# ============================= Parser =============================
class Parser:
    def __init__(self,toks): self.toks=toks; self.i=0
    def cur(self): return self.toks[self.i]
    def at(self,k): return self.cur().k==k
    def eat(self,k,msg=None):
        if not self.at(k): t=self.cur(); raise SyntaxError(msg or f'Expected {k}, got {t.k} at {t.l}:{t.c}')
        tok=self.cur(); self.i+=1; return tok
    def opt(self,k): 
        if self.at(k): return self.eat(k)
        return None

    def parse(self):
        ds=[]
        while not self.at('EOF'):
            if self.at('extern'): ds.append(self.parse_extern())
            else: ds.append(self.parse_func())
        return Program(ds)

    def parse_type(self):
        if self.at('int') or self.at('double') or self.at('bool') or self.at('string') or self.at('void'):
            return type_from_name(self.eat(self.cur().k).v)
        raise SyntaxError(f'Expected type at {self.cur().l}:{self.cur().c}')

    def parse_params(self):
        ps=[]
        if self.at('RP'): return ps
        while True:
            name=self.eat('ID').v
            self.eat('COLON'); ty=self.parse_type()
            ps.append(Param(name,ty))
            if self.opt('COMMA'): continue
            break
        return ps

    def parse_extern(self):
        self.eat('extern'); self.eat('fn'); name=self.eat('ID').v
        self.eat('LP'); ps=self.parse_params(); self.eat('RP')
        self.eat('ARROW'); ret=self.parse_type()
        self.eat('SEMI')
        return Extern(name,ps,ret)

    def parse_func(self):
        self.eat('fn'); name=self.eat('ID').v
        self.eat('LP'); ps=self.parse_params(); self.eat('RP')
        ret=T_VOID
        if self.opt('ARROW'): ret=self.parse_type()
        body=self.parse_block()
        return Func(name,ps,ret,body)

    def parse_block(self):
        self.eat('LB'); ss=[]
        while not self.at('RB'): ss.append(self.parse_stmt())
        self.eat('RB'); return Block(ss)

    def parse_stmt(self):
        if self.at('let'):
            self.eat('let'); name=self.eat('ID').v
            self.eat('COLON'); ty=self.parse_type()
            expr=None
            if self.opt('ASSIGN'): expr=self.parse_expr()
            self.eat('SEMI'); return Let(name,ty,expr)
        if self.at('return'):
            self.eat('return'); e=None
            if not self.at('SEMI'): e=self.parse_expr()
            self.eat('SEMI'); return Return(e)
        if self.at('if'):
            self.eat('if'); self.eat('LP'); cond=self.parse_expr(); self.eat('RP')
            then=self.parse_block(); els=None
            if self.opt('else'): els=self.parse_block()
            return If(cond,then,els)
        if self.at('while'):
            self.eat('while'); self.eat('LP'); cond=self.parse_expr(); self.eat('RP')
            body=self.parse_block(); return While(cond,body)
        if self.at('print'):
            self.eat('print'); self.eat('LP')
            args=[]
            if not self.at('RP'):
                while True:
                    args.append(self.parse_expr())
                    if self.opt('COMMA'): continue
                    break
            self.eat('RP'); self.eat('SEMI')
            return Print(args)
        if self.at('ID') and self.toks[self.i+1].k=='ASSIGN':
            name=self.eat('ID').v; self.eat('ASSIGN'); e=self.parse_expr(); self.eat('SEMI')
            return Assign(name,e)
        e=self.parse_expr(); self.eat('SEMI'); return ExprStmt(e)

    # precedence
    def parse_expr(self): return self.parse_or()
    def parse_or(self):
        e=self.parse_and()
        while self.opt('OR'): e=Binary('||',e,self.parse_and())
        return e
    def parse_and(self):
        e=self.parse_eq()
        while self.opt('AND'): e=Binary('&&',e,self.parse_eq())
        return e
    def parse_eq(self):
        e=self.parse_rel()
        while True:
            if self.opt('EQ'): e=Binary('==',e,self.parse_rel())
            elif self.opt('NE'): e=Binary('!=',e,self.parse_rel())
            else: break
        return e
    def parse_rel(self):
        e=self.parse_add()
        while True:
            if self.opt('LE'): e=Binary('<=',e,self.parse_add())
            elif self.opt('GE'): e=Binary('>=',e,self.parse_add())
            elif self.opt('LT'): e=Binary('<',e,self.parse_add())
            elif self.opt('GT'): e=Binary('>',e,self.parse_add())
            else: break
        return e
    def parse_add(self):
        e=self.parse_mul()
        while True:
            if self.opt('PLUS'): e=Binary('+',e,self.parse_mul())
            elif self.opt('MINUS'): e=Binary('-',e,self.parse_mul())
            else: break
        return e
    def parse_mul(self):
        e=self.parse_un()
        while True:
            if self.opt('STAR'): e=Binary('*',e,self.parse_un())
            elif self.opt('SLASH'): e=Binary('/',e,self.parse_un())
            elif self.opt('PERCENT'): e=Binary('%',e,self.parse_un())
            else: break
        return e
    def parse_un(self):
        if self.opt('MINUS'): return Unary('-', self.parse_un())
        if self.opt('BANG'):  return Unary('!', self.parse_un())
        return self.parse_call()
    def parse_call(self):
        e=self.parse_primary()
        while self.at('LP'):
            self.eat('LP'); args=[]
            if not self.at('RP'):
                while True:
                    args.append(self.parse_expr())
                    if self.opt('COMMA'): continue
                    break
            self.eat('RP')
            if isinstance(e,Var): e=Call(e.name,args)
            else: raise SyntaxError(f'Call target must be a name at {self.cur().l}:{self.cur().c}')
        return e
    def parse_primary(self):
        t=self.cur()
        if self.opt('INT'):    return Lit(t.v,T_INT)
        if self.opt('FLOAT'):  return Lit(t.v,T_DBL)
        if self.opt('STRING'): return Lit(t.v,T_STR)
        if self.opt('true'):   return Lit(1,T_BOOL)
        if self.opt('false'):  return Lit(0,T_BOOL)
        if self.opt('ID'):     return Var(t.v)
        if self.opt('LP'):
            e=self.parse_expr(); self.eat('RP'); return e
        raise SyntaxError(f'Unexpected token {t.k} at {t.l}:{t.c}')

# ============================ Semantics ============================
class FuncSig:
    def __init__(self,name,ps,ret): self.name=name; self.param_types=ps; self.ret=ret
class Ctx:
    def __init__(self):
        self.funcs={}; self.vars=[]; self.cur=None
    def push(self): self.vars.append({})
    def pop(self): self.vars.pop()
    def decl_var(self,n,t):
        if n in self.vars[-1]: raise TypeError(f'Variable {n} already declared')
        self.vars[-1][n]=t
    def get_var(self,n):
        for d in reversed(self.vars):
            if n in d: return d[n]
        raise TypeError(f'Undeclared variable {n}')
    def decl_fn(self,name,sig): self.funcs[name]=sig
    def get_fn(self,name):
        if name in self.funcs: return self.funcs[name]
        raise TypeError(f'Undeclared function {name}')

def is_num(t): return t in (T_INT,T_DBL)
def can_assign(dst,src): return dst==src or (dst==T_DBL and src==T_INT)

def analyze_expr(e,ctx:Ctx):
    if isinstance(e,Lit): return e.ty
    if isinstance(e,Var): e.ty=ctx.get_var(e.name); return e.ty
    if isinstance(e,Unary):
        rt=analyze_expr(e.rhs,ctx)
        if e.op=='-':
            if not is_num(rt): raise TypeError('Unary - expects numeric'); e.ty=rt
            return rt
        if e.op=='!':
            if rt not in (T_BOOL,T_INT,T_DBL): raise TypeError('! expects bool/numeric')
            e.ty=T_BOOL; return T_BOOL
    if isinstance(e,Binary):
        lt=analyze_expr(e.lhs,ctx); rt=analyze_expr(e.rhs,ctx); op=e.op
        if op in ('*','/','+','-'):
            if not (is_num(lt) and is_num(rt)): raise TypeError(f'Op {op} needs numeric')
            e.ty = T_DBL if (lt==T_DBL or rt==T_DBL) else T_INT; return e.ty
        if op=='%':
            if lt!=T_INT or rt!=T_INT: raise TypeError('% requires int')
            e.ty=T_INT; return T_INT
        if op in ('<','<=','>','>='):
            if not (is_num(lt) and is_num(rt)): raise TypeError('Comparison needs numeric')
            e.ty=T_BOOL; return T_BOOL
        if op in ('==','!='):
            if lt!=rt and not (is_num(lt) and is_num(rt)): raise TypeError('==/!= need same or both numeric')
            e.ty=T_BOOL; return T_BOOL
        if op in ('&&','||'):
            if lt not in (T_BOOL,T_INT,T_DBL) or rt not in (T_BOOL,T_INT,T_DBL):
                raise TypeError('Logical ops need bool/numeric')
            e.ty=T_BOOL; return T_BOOL
    if isinstance(e,Call):
        sig=ctx.get_fn(e.name)
        if len(sig.param_types)!=len(e.args): raise TypeError(f'Arity mismatch for {e.name}')
        for i,(pt,a) in enumerate(zip(sig.param_types,e.args),1):
            at=analyze_expr(a,ctx)
            if pt==at: continue
            if pt==T_DBL and at==T_INT: continue
            raise TypeError(f'Arg {i} to {e.name} expects {pt}, got {at}')
        e.ty=sig.ret; return e.ty
    raise TypeError('Unknown expression')

def analyze_stmt(s,ctx:Ctx):
    if isinstance(s,Let):
        if s.expr:
            et=analyze_expr(s.expr,ctx)
            if not can_assign(s.ty,et): raise TypeError(f'Cannot assign {et} to {s.ty}')
        ctx.decl_var(s.name,s.ty)
    elif isinstance(s,Assign):
        vt=ctx.get_var(s.name); et=analyze_expr(s.expr,ctx)
        if not can_assign(vt,et): raise TypeError(f'Cannot assign {et} to {vt}')
    elif isinstance(s,If):
        analyze_expr(s.cond,ctx)
        ctx.push(); [analyze_stmt(x,ctx) for x in s.then.stmts]; ctx.pop()
        if s.els: ctx.push(); [analyze_stmt(x,ctx) for x in s.els.stmts]; ctx.pop()
    elif isinstance(s,While):
        analyze_expr(s.cond,ctx)
        ctx.push(); [analyze_stmt(x,ctx) for x in s.body.stmts]; ctx.pop()
    elif isinstance(s,Return):
        if s.expr is None:
            if ctx.cur.ret!=T_VOID and ctx.cur.name!='main': raise TypeError('Return without value')
        else:
            rt=analyze_expr(s.expr,ctx)
            if ctx.cur.ret==T_DBL and rt==T_INT: pass
            elif ctx.cur.ret!=rt: raise TypeError(f'Return type {rt} != {ctx.cur.ret}')
    elif isinstance(s,Print):
        for a in s.args: analyze_expr(a,ctx)
    elif isinstance(s,ExprStmt):
        if not isinstance(s.expr,Call): raise TypeError('Only function calls allowed as expression statements')
        analyze_expr(s.expr,ctx)

def analyze(prog:Program):
    ctx=Ctx()
    for d in prog.decls:
        if isinstance(d,Extern) or isinstance(d,Func):
            ctx.decl_fn(d.name, FuncSig(d.name,[p.ty for p in d.params], d.ret))
    for d in prog.decls:
        if isinstance(d,Func):
            ctx.cur = ctx.get_fn(d.name)
            ctx.push()
            for p in d.params: ctx.decl_var(p.name,p.ty)
            for st in d.body.stmts: analyze_stmt(st,ctx)
            ctx.pop()
    if 'main' not in ctx.funcs:
        print('// Warning: no main() defined; producing a library C file.', file=sys.stderr)
    return ctx

# ============================= Codegen =============================
def c_type(t:Type):
    if t==T_INT: return 'long long'
    if t==T_DBL: return 'double'
    if t==T_BOOL: return 'bool'
    if t==T_STR: return 'const char*'
    if t==T_VOID: return 'void'
    raise RuntimeError('unknown type')

def c_escape(s:str):
    return s.replace('\\','\\\\').replace('"','\\"').replace('\n','\\n').replace('\t','\\t')

class CGen:
    def __init__(self,prog:Program,ctx:Ctx):
        self.p=prog; self.ctx=ctx; self.out=[]; self.ind=0
    def emit(self,s=''): self.out.append('  '*self.ind + s)
    def gen(self):
        self.emit('#include <stdio.h>')
        self.emit('#include <math.h>')
        self.emit('#include <stdbool.h>')
        self.emit('#include <string.h>')
        self.emit('')
        self.emit('static inline void __p_int(long long x){ printf("%lld", x); }')
        self.emit('static inline void __p_double(double x){ printf("%.15g", x); }')
        self.emit('static inline void __p_bool(bool x){ printf("%s", x ? "true" : "false"); }')
        self.emit('static inline void __p_string(const char* s){ printf("%s", s); }')
        self.emit('')
        for d in self.p.decls:
            if isinstance(d,Extern): self.emit(self.sig_to_c(d.name,d.params,d.ret) + ';')
        for d in self.p.decls:
            if isinstance(d,Func): self.emit(self.sig_to_c(d.name,d.params,d.ret) + ';')
        self.emit('')
        for d in self.p.decls:
            if isinstance(d,Func): self.gen_func(d)
        return '\n'.join(self.out)
    def sig_to_c(self,name,ps,ret):
        if name=='main': return 'int main(void)'
        params = ', '.join(f'{c_type(p.ty)} {p.name}' for p in ps) or 'void'
        return f'{c_type(ret)} {name}({params})'
    def gen_func(self,f:Func):
        self.emit(self.sig_to_c(f.name,f.params,f.ret) + ' {'); self.ind+=1
        for st in f.body.stmts: self.gen_stmt(st)
        if f.name=='main': self.emit('return 0;')
        self.ind-=1; self.emit('}'); self.emit('')
    def gen_stmt(self,s):
        if isinstance(s,Let):
            init = f' = {self.gen_expr(s.expr)}' if s.expr is not None else ''
            self.emit(f'{c_type(s.ty)} {s.name}{init};')
        elif isinstance(s,Assign):
            self.emit(f'{s.name} = {self.gen_expr(s.expr)};')
        elif isinstance(s,If):
            self.emit(f'if ({self.gen_expr(s.cond)}) {{'); self.ind+=1
            for x in s.then.stmts: self.gen_stmt(x)
            self.ind-=1; self.emit('}')
            if s.els:
                self.emit('else {'); self.ind+=1
                for x in s.els.stmts: self.gen_stmt(x)
                self.ind-=1; self.emit('}')
        elif isinstance(s,While):
            self.emit(f'while ({self.gen_expr(s.cond)}) {{'); self.ind+=1
            for x in s.body.stmts: self.gen_stmt(x)
            self.ind-=1; self.emit('}')
        elif isinstance(s,Return):
            if s.expr is None: self.emit('return;')
            else: self.emit(f'return {self.gen_expr(s.expr)};')
        elif isinstance(s,Print):
            first=True
            for a in s.args:
                if not first: self.emit('printf(" ");')
                first=False
                ty=self.expr_static_type(a); ce=self.gen_expr(a)
                if   ty==T_INT:   self.emit(f'__p_int({ce});')
                elif ty==T_DBL:   self.emit(f'__p_double({ce});')
                elif ty==T_BOOL:  self.emit(f'__p_bool({ce});')
                elif ty==T_STR:   self.emit(f'__p_string({ce});')
                else:              self.emit(f'__p_string("<?>");')
            self.emit('printf("\\n");')
        elif isinstance(s,ExprStmt):
            self.emit(self.gen_expr(s.expr) + ';')
    def expr_static_type(self,e):
        if isinstance(e,Lit): return e.ty
        if isinstance(e,Call): return self.ctx.get_fn(e.name).ret
        if isinstance(e,Unary):
            t=self.expr_static_type(e.rhs); return T_BOOL if e.op=='!' else t
        if isinstance(e,Binary):
            if e.op in ('*','/','+','-'): return T_DBL
            if e.op=='%': return T_INT
            return T_BOOL
        return None
    def gen_expr(self,e):
        if isinstance(e,Lit):
            if e.ty==T_INT: return str(int(e.val))
            if e.ty==T_DBL:
                s=repr(float(e.val))
                if '.' not in s and 'e' not in s and 'E' not in s: s+='.0'
                return s
            if e.ty==T_BOOL: return 'true' if e.val else 'false'
            if e.ty==T_STR:  return f'"{c_escape(e.val)}"'
        if isinstance(e,Var): return e.name
        if isinstance(e,Unary): return f'({e.op}{self.gen_expr(e.rhs)})'
        if isinstance(e,Binary): return f'({self.gen_expr(e.lhs)} {e.op} {self.gen_expr(e.rhs)})'
        if isinstance(e,Call):
            args=', '.join(self.gen_expr(a) for a in e.args)
            return f'{e.name}({args})'
        raise RuntimeError('unknown expr')

# ============================== Driver =============================
def compile_source(src:str)->str:
    toks=lex(src)
    ast=Parser(toks).parse()
    ctx=analyze(ast)
    return CGen(ast,ctx).gen()

def main():
    if len(sys.argv)!=2:
        print('Usage: echc.py <input.ech>  # outputs C on stdout', file=sys.stderr)
        sys.exit(1)
    src=open(sys.argv[1],'r',encoding='utf-8').read()
    try:
        sys.stdout.write(compile_source(src))
    except (SyntaxError,TypeError) as e:
        print(f'Compile error: {e}', file=sys.stderr)
        sys.exit(2)

if __name__=='__main__':
    main()

Then:

python3 echc.py hello.ech > hello.c
cc hello.c -o hello -lm
./hello
# -> Hello, world!

If you still hit an error, paste the exact hello.ech you’re using and I’ll adapt the grammar to match it.
You said:
(base) ian-martin-ajzenszmidt@ian-martin-ajzenszmidt-CFSZ5-3:~/Downloads$ python3 echc.py hello.ech > hello.c
cc hello.c -o hello -lm
./hello
# -> Hello, world!
  File "/home/ian-martin-ajzenszmidt/Downloads/echc.py", line 117
    class Program(Node):  def __init__(self,decls): self.decls=decls
                          ^^^
SyntaxError: invalid syntax
/usr/bin/ld: /usr/lib/gcc/x86_64-linux-gnu/13/../../../x86_64-linux-gnu/Scrt1.o: in function _start':
(.text+0x1b): undefined reference to main'
collect2: error: ld returned 1 exit status
bash: ./hello: No such file or directory
(
ChatGPT said:

You hit a Python syntax error because I (oops) inlined def inside class headers. Replace your echc.py with this fixed single-file compiler:

#!/usr/bin/env python3
# echc.py — Echidna: tiny single-file compiler from .ech -> portable C
# Usage:
#   python3 echc.py hello.ech > hello.c
#   cc hello.c -o hello -lm
# Language:
#   Types: int, double, bool, string, void
#   Decls: extern fn f(...)->T;  fn f(...)->T { ... }
#   Stmts: let name: T (= expr)? ;  name = expr;
#          if (expr) { ... } else { ... }
#          while (expr) { ... }
#          return expr? ;
#          print(expr[, expr...]);   # space-separated, newline
#          expr ;   # call-only stmt
#   Expr: literals (123, 1.23, true, false, "str"), vars, calls
#         unary: - ! ;  binary: * / % + - < <= > >= == != && ||

import sys

# ============================== Lexing ==============================
TT = {
  'EOF':'EOF','ID':'ID','INT':'INT','FLOAT':'FLOAT','STRING':'STRING',
  'LP':'(','RP':')','LB':'{','RB':'}','COMMA':',','SEMI':';','COLON':':',
  'ARROW':'->','ASSIGN':'=','PLUS':'+','MINUS':'-','STAR':'*','SLASH':'/',
  'PERCENT':'%','BANG':'!','LT':'<','LE':'<=','GT':'>','GE':'>=',
  'EQ':'==','NE':'!=','AND':'&&','OR':'||',
}
KEYWORDS = {'fn','extern','let','return','if','else','while',
            'true','false','int','double','bool','string','void','print'}

class Tok:
    def __init__(self,k,v,l,c):
        self.k=k; self.v=v; self.l=l; self.c=c

def lex(src:str):
    i,n,l,c = 0,len(src),1,1
    def peek(k=0):
        j=i+k
        return src[j] if j<n else ''
    def adv():
        nonlocal i,l,c
        ch = peek()
        if ch=='\n': l+=1; c=1
        else: c+=1
        i+=1
        return ch
    def match(s):
        nonlocal i
        if src.startswith(s,i):
            for _ in s: adv()
            return True
        return False
    toks=[]
    while True:
        ch = peek()
        if ch=='':
            toks.append(Tok('EOF','',l,c)); break
        if ch in ' \t\r\n': adv(); continue
        # comments: //... and #...
        if (ch=='/' and peek(1)=='/') or ch=='#':
            while peek() and peek()!='\n': adv()
            continue
        # multi-char ops
        if   match('->'): toks.append(Tok('ARROW','->',l,c)); continue
        if   match('<='): toks.append(Tok('LE','<=',l,c)); continue
        if   match('>='): toks.append(Tok('GE','>=',l,c)); continue
        if   match('=='): toks.append(Tok('EQ','==',l,c)); continue
        if   match('!='): toks.append(Tok('NE','!=',l,c)); continue
        if   match('&&'): toks.append(Tok('AND','&&',l,c)); continue
        if   match('||'): toks.append(Tok('OR','||',l,c)); continue
        # string
        if ch=='"':
            sl,sc=l,c; adv()
            buf=[]
            while True:
                p=peek()
                if p=='': raise SyntaxError(f'Unterminated string at {sl}:{sc}')
                if p=='"': adv(); break
                if p=='\\':
                    adv(); q=peek()
                    if   q=='n': buf.append('\n'); adv()
                    elif q=='t': buf.append('\t'); adv()
                    elif q=='"': buf.append('"');  adv()
                    elif q=='\\': buf.append('\\'); adv()
                    else: raise SyntaxError(f'Bad escape \\{q} at {l}:{c}')
                else:
                    buf.append(adv())
            toks.append(Tok('STRING',''.join(buf),sl,sc)); continue
        # number
        if ch.isdigit():
            sl,sc=l,c; s=adv(); isf=False
            while peek().isdigit(): s+=adv()
            if peek()=='.' and peek(1).isdigit():
                isf=True; s+=adv()
                while peek().isdigit(): s+=adv()
            toks.append(Tok('FLOAT' if isf else 'INT', float(s) if isf else int(s), sl, sc)); continue
        # id / keyword
        if ch.isalpha() or ch=='_':
            sl,sc=l,c; s=adv()
            while peek().isalnum() or peek()=='_': s+=adv()
            toks.append(Tok(s if s in KEYWORDS else 'ID', s, sl, sc)); continue
        # single-char
        single = {'(':'LP',')':'RP','{':'LB','}':'RB',',':'COMMA',';':'SEMI',':':'COLON',
                  '+':'PLUS','-':'MINUS','*':'STAR','/':'SLASH','%':'PERCENT',
                  '!':'BANG','<':'LT','>':'GT','=':'ASSIGN'}
        if ch in single:
            k = single[ch]; adv(); toks.append(Tok(k,ch,l,c)); continue
        raise SyntaxError(f'Lex error near {src[i:i+20]!r}')
    return toks

# ========================== AST & Types ============================
class Type:
    def __init__(self,k): self.k=k
    def __eq__(self,o): return isinstance(o,Type) and self.k==o.k
    def __repr__(self): return self.k
T_INT=Type('int'); T_DBL=Type('double'); T_BOOL=Type('bool'); T_STR=Type('string'); T_VOID=Type('void')
def type_from_name(n): return {'int':T_INT,'double':T_DBL,'bool':T_BOOL,'string':T_STR,'void':T_VOID}[n]

class Node: pass

class Program(Node):
    def __init__(self,decls): self.decls=decls

class Extern(Node):
    def __init__(self,name,params,ret): self.name=name; self.params=params; self.ret=ret

class Func(Node):
    def __init__(self,name,params,ret,body): self.name=name; self.params=params; self.ret=ret; self.body=body

class Param(Node):
    def __init__(self,name,ty): self.name=name; self.ty=ty

class Block(Node):
    def __init__(self,stmts): self.stmts=stmts

class Let(Node):
    def __init__(self,name,ty,expr): self.name=name; self.ty=ty; self.expr=expr

class Assign(Node):
    def __init__(self,name,expr): self.name=name; self.expr=expr

class If(Node):
    def __init__(self,cond,then,els): self.cond=cond; self.then=then; self.els=els

class While(Node):
    def __init__(self,cond,body): self.cond=cond; self.body=body

class Return(Node):
    def __init__(self,expr): self.expr=expr

class Print(Node):
    def __init__(self,args): self.args=args

class ExprStmt(Node):
    def __init__(self,expr): self.expr=expr

# Exprs
class Var(Node):
    def __init__(self,name): self.name=name; self.ty=None

class Lit(Node):
    def __init__(self,val,ty): self.val=val; self.ty=ty

class Call(Node):
    def __init__(self,name,args): self.name=name; self.args=args; self.ty=None

class Unary(Node):
    def __init__(self,op,rhs): self.op=op; self.rhs=rhs; self.ty=None

class Binary(Node):
    def __init__(self,op,lhs,rhs): self.op=op; self.lhs=lhs; self.rhs=rhs; self.ty=None

# ============================= Parser =============================
class Parser:
    def __init__(self,toks): self.toks=toks; self.i=0
    def cur(self): return self.toks[self.i]
    def at(self,k): return self.cur().k==k
    def eat(self,k,msg=None):
        if not self.at(k):
            t=self.cur(); raise SyntaxError(msg or f'Expected {k}, got {t.k} at {t.l}:{t.c}')
        tok=self.cur(); self.i+=1; return tok
    def opt(self,k):
        if self.at(k): return self.eat(k)
        return None

    def parse(self):
        ds=[]
        while not self.at('EOF'):
            if self.at('extern'): ds.append(self.parse_extern())
            else: ds.append(self.parse_func())
        return Program(ds)

    def parse_type(self):
        if self.at('int') or self.at('double') or self.at('bool') or self.at('string') or self.at('void'):
            return type_from_name(self.eat(self.cur().k).v)
        raise SyntaxError(f'Expected type at {self.cur().l}:{self.cur().c}')

    def parse_params(self):
        ps=[]
        if self.at('RP'): return ps
        while True:
            name=self.eat('ID').v
            self.eat('COLON'); ty=self.parse_type()
            ps.append(Param(name,ty))
            if self.opt('COMMA'): continue
            break
        return ps

    def parse_extern(self):
        self.eat('extern'); self.eat('fn'); name=self.eat('ID').v
        self.eat('LP'); ps=self.parse_params(); self.eat('RP')
        self.eat('ARROW'); ret=self.parse_type()
        self.eat('SEMI')
        return Extern(name,ps,ret)

    def parse_func(self):
        self.eat('fn'); name=self.eat('ID').v
        self.eat('LP'); ps=self.parse_params(); self.eat('RP')
        ret=T_VOID
        if self.opt('ARROW'): ret=self.parse_type()
        body=self.parse_block()
        return Func(name,ps,ret,body)

    def parse_block(self):
        self.eat('LB'); ss=[]
        while not self.at('RB'): ss.append(self.parse_stmt())
        self.eat('RB'); return Block(ss)

    def parse_stmt(self):
        if self.at('let'):
            self.eat('let'); name=self.eat('ID').v
            self.eat('COLON'); ty=self.parse_type()
            expr=None
            if self.opt('ASSIGN'): expr=self.parse_expr()
            self.eat('SEMI'); return Let(name,ty,expr)
        if self.at('return'):
            self.eat('return'); e=None
            if not self.at('SEMI'): e=self.parse_expr()
            self.eat('SEMI'); return Return(e)
        if self.at('if'):
            self.eat('if'); self.eat('LP'); cond=self.parse_expr(); self.eat('RP')
            then=self.parse_block(); els=None
            if self.opt('else'): els=self.parse_block()
            return If(cond,then,els)
        if self.at('while'):
            self.eat('while'); self.eat('LP'); cond=self.parse_expr(); self.eat('RP')
            body=self.parse_block(); return While(cond,body)
        if self.at('print'):
            self.eat('print'); self.eat('LP')
            args=[]
            if not self.at('RP'):
                while True:
                    args.append(self.parse_expr())
                    if self.opt('COMMA'): continue
                    break
            self.eat('RP'); self.eat('SEMI')
            return Print(args)
        if self.at('ID') and self.toks[self.i+1].k=='ASSIGN':
            name=self.eat('ID').v; self.eat('ASSIGN'); e=self.parse_expr(); self.eat('SEMI')
            return Assign(name,e)
        e=self.parse_expr(); self.eat('SEMI'); return ExprStmt(e)

    # precedence
    def parse_expr(self): return self.parse_or()
    def parse_or(self):
        e=self.parse_and()
        while self.opt('OR'): e=Binary('||',e,self.parse_and())
        return e
    def parse_and(self):
        e=self.parse_eq()
        while self.opt('AND'): e=Binary('&&',e,self.parse_eq())
        return e
    def parse_eq(self):
        e=self.parse_rel()
        while True:
            if self.opt('EQ'): e=Binary('==',e,self.parse_rel())
            elif self.opt('NE'): e=Binary('!=',e,self.parse_rel())
            else: break
        return e
    def parse_rel(self):
        e=self.parse_add()
        while True:
            if self.opt('LE'): e=Binary('<=',e,self.parse_add())
            elif self.opt('GE'): e=Binary('>=',e,self.parse_add())
            elif self.opt('LT'): e=Binary('<',e,self.parse_add())
            elif self.opt('GT'): e=Binary('>',e,self.parse_add())
            else: break
        return e
    def parse_add(self):
        e=self.parse_mul()
        while True:
            if self.opt('PLUS'): e=Binary('+',e,self.parse_mul())
            elif self.opt('MINUS'): e=Binary('-',e,self.parse_mul())
            else: break
        return e
    def parse_mul(self):
        e=self.parse_un()
        while True:
            if self.opt('STAR'): e=Binary('*',e,self.parse_un())
            elif self.opt('SLASH'): e=Binary('/',e,self.parse_un())
            elif self.opt('PERCENT'): e=Binary('%',e,self.parse_un())
            else: break
        return e
    def parse_un(self):
        if self.opt('MINUS'): return Unary('-', self.parse_un())
        if self.opt('BANG'):  return Unary('!', self.parse_un())
        return self.parse_call()
    def parse_call(self):
        e=self.parse_primary()
        while self.at('LP'):
            self.eat('LP'); args=[]
            if not self.at('RP'):
                while True:
                    args.append(self.parse_expr())
                    if self.opt('COMMA'): continue
                    break
            self.eat('RP')
            if isinstance(e,Var): e=Call(e.name,args)
            else: raise SyntaxError(f'Call target must be a name at {self.cur().l}:{self.cur().c}')
        return e
    def parse_primary(self):
        t=self.cur()
        if self.opt('INT'):    return Lit(t.v,T_INT)
        if self.opt('FLOAT'):  return Lit(t.v,T_DBL)
        if self.opt('STRING'): return Lit(t.v,T_STR)
        if self.opt('true'):   return Lit(1,T_BOOL)
        if self.opt('false'):  return Lit(0,T_BOOL)
        if self.opt('ID'):     return Var(t.v)
        if self.opt('LP'):
            e=self.parse_expr(); self.eat('RP'); return e
        raise SyntaxError(f'Unexpected token {t.k} at {t.l}:{t.c}')

# ============================ Semantics ============================
class FuncSig:
    def __init__(self,name,ps,ret):
        self.name=name; self.param_types=ps; self.ret=ret

class Ctx:
    def __init__(self):
        self.funcs={}; self.vars=[]; self.cur=None
    def push(self): self.vars.append({})
    def pop(self): self.vars.pop()
    def decl_var(self,n,t):
        if n in self.vars[-1]: raise TypeError(f'Variable {n} already declared')
        self.vars[-1][n]=t
    def get_var(self,n):
        for d in reversed(self.vars):
            if n in d: return d[n]
        raise TypeError(f'Undeclared variable {n}')
    def decl_fn(self,name,sig): self.funcs[name]=sig
    def get_fn(self,name):
        if name in self.funcs: return self.funcs[name]
        raise TypeError(f'Undeclared function {name}')

def is_num(t): return t in (T_INT,T_DBL)
def can_assign(dst,src): return dst==src or (dst==T_DBL and src==T_INT)

def analyze_expr(e,ctx:Ctx):
    if isinstance(e,Lit): return e.ty
    if isinstance(e,Var):
        e.ty=ctx.get_var(e.name); return e.ty
    if isinstance(e,Unary):
        rt=analyze_expr(e.rhs,ctx)
        if e.op=='-':
            if not is_num(rt): raise TypeError('Unary - expects numeric')
            e.ty=rt; return rt
        if e.op=='!':
            if rt not in (T_BOOL,T_INT,T_DBL): raise TypeError('! expects bool/numeric')
            e.ty=T_BOOL; return T_BOOL
    if isinstance(e,Binary):
        lt=analyze_expr(e.lhs,ctx); rt=analyze_expr(e.rhs,ctx); op=e.op
        if op in ('*','/','+','-'):
            if not (is_num(lt) and is_num(rt)): raise TypeError(f'Op {op} needs numeric')
            e.ty = T_DBL if (lt==T_DBL or rt==T_DBL) else T_INT; return e.ty
        if op=='%':
            if lt!=T_INT or rt!=T_INT: raise TypeError('% requires int')
            e.ty=T_INT; return T_INT
        if op in ('<','<=','>','>='):
            if not (is_num(lt) and is_num(rt)): raise TypeError('Comparison needs numeric')
            e.ty=T_BOOL; return T_BOOL
        if op in ('==','!='):
            if lt!=rt and not (is_num(lt) and is_num(rt)): raise TypeError('==/!= need same or both numeric')
            e.ty=T_BOOL; return T_BOOL
        if op in ('&&','||'):
            if lt not in (T_BOOL,T_INT,T_DBL) or rt not in (T_BOOL,T_INT,T_DBL):
                raise TypeError('Logical ops need bool/numeric')
            e.ty=T_BOOL; return T_BOOL
    if isinstance(e,Call):
        sig=ctx.get_fn(e.name)
        if len(sig.param_types)!=len(e.args): raise TypeError(f'Arity mismatch for {e.name}')
        for i,(pt,a) in enumerate(zip(sig.param_types,e.args),1):
            at=analyze_expr(a,ctx)
            if pt==at: continue
            if pt==T_DBL and at==T_INT: continue
            raise TypeError(f'Arg {i} to {e.name} expects {pt}, got {at}')
        e.ty=sig.ret; return e.ty
    raise TypeError('Unknown expression')

def analyze_stmt(s,ctx:Ctx):
    if isinstance(s,Let):
        if s.expr:
            et=analyze_expr(s.expr,ctx)
            if not can_assign(s.ty,et): raise TypeError(f'Cannot assign {et} to {s.ty}')
        ctx.decl_var(s.name,s.ty)
    elif isinstance(s,Assign):
        vt=ctx.get_var(s.name); et=analyze_expr(s.expr,ctx)
        if not can_assign(vt,et): raise TypeError(f'Cannot assign {et} to {vt}')
    elif isinstance(s,If):
        analyze_expr(s.cond,ctx)
        ctx.push(); [analyze_stmt(x,ctx) for x in s.then.stmts]; ctx.pop()
        if s.els: ctx.push(); [analyze_stmt(x,ctx) for x in s.els.stmts]; ctx.pop()
    elif isinstance(s,While):
        analyze_expr(s.cond,ctx)
        ctx.push(); [analyze_stmt(x,ctx) for x in s.body.stmts]; ctx.pop()
    elif isinstance(s,Return):
        if s.expr is None:
            if ctx.cur.ret!=T_VOID and ctx.cur.name!='main': raise TypeError('Return without value')
        else:
            rt=analyze_expr(s.expr,ctx)
            if ctx.cur.ret==T_DBL and rt==T_INT: pass
            elif ctx.cur.ret!=rt: raise TypeError(f'Return type {rt} != {ctx.cur.ret}')
    elif isinstance(s,Print):
        for a in s.args: analyze_expr(a,ctx)
    elif isinstance(s,ExprStmt):
        if not isinstance(s.expr,Call): raise TypeError('Only function calls allowed as expression statements')
        analyze_expr(s.expr,ctx)

def analyze(prog:Program):
    ctx=Ctx()
    for d in prog.decls:
        if isinstance(d,Extern) or isinstance(d,Func):
            ctx.decl_fn(d.name, FuncSig(d.name,[p.ty for p in d.params], d.ret))
    for d in prog.decls:
        if isinstance(d,Func):
            ctx.cur = ctx.get_fn(d.name)
            ctx.push()
            for p in d.params: ctx.decl_var(p.name,p.ty)
            for st in d.body.stmts: analyze_stmt(st,ctx)
            ctx.pop()
    if 'main' not in ctx.funcs:
        print('// Warning: no main() defined; producing a library C file.', file=sys.stderr)
    return ctx

# ============================= Codegen =============================
def c_type(t:Type):
    if t==T_INT: return 'long long'
    if t==T_DBL: return 'double'
    if t==T_BOOL: return 'bool'
    if t==T_STR: return 'const char*'
    if t==T_VOID: return 'void'
    raise RuntimeError('unknown type')

def c_escape(s:str):
    return s.replace('\\','\\\\').replace('"','\\"').replace('\n','\\n').replace('\t','\\t')

class CGen:
    def __init__(self,prog:Program,ctx:Ctx):
        self.p=prog; self.ctx=ctx; self.out=[]; self.ind=0
    def emit(self,s=''): self.out.append('  '*self.ind + s)
    def gen(self):
        self.emit('#include <stdio.h>')
        self.emit('#include <math.h>')
        self.emit('#include <stdbool.h>')
        self.emit('#include <string.h>')
        self.emit('')
        self.emit('static inline void __p_int(long long x){ printf("%lld", x); }')
        self.emit('static inline void __p_double(double x){ printf("%.15g", x); }')
        self.emit('static inline void __p_bool(bool x){ printf("%s", x ? "true" : "false"); }')
        self.emit('static inline void __p_string(const char* s){ printf("%s", s); }')
        self.emit('')
        for d in self.p.decls:
            if isinstance(d,Extern): self.emit(self.sig_to_c(d.name,d.params,d.ret) + ';')
        for d in self.p.decls:
            if isinstance(d,Func): self.emit(self.sig_to_c(d.name,d.params,d.ret) + ';')
        self.emit('')
        for d in self.p.decls:
            if isinstance(d,Func): self.gen_func(d)
        return '\n'.join(self.out)
    def sig_to_c(self,name,ps,ret):
        if name=='main': return 'int main(void)'
        params = ', '.join(f'{c_type(p.ty)} {p.name}' for p in ps) or 'void'
        return f'{c_type(ret)} {name}({params})'
    def gen_func(self,f:Func):
        self.emit(self.sig_to_c(f.name,f.params,f.ret) + ' {'); self.ind+=1
        for st in f.body.stmts: self.gen_stmt(st)
        if f.name=='main': self.emit('return 0;')
        self.ind-=1; self.emit('}'); self.emit('')
    def gen_stmt(self,s):
        if isinstance(s,Let):
            init = f' = {self.gen_expr(s.expr)}' if s.expr is not None else ''
            self.emit(f'{c_type(s.ty)} {s.name}{init};')
        elif isinstance(s,Assign):
            self.emit(f'{s.name} = {self.gen_expr(s.expr)};')
        elif isinstance(s,If):
            self.emit(f'if ({self.gen_expr(s.cond)}) {{'); self.ind+=1
            for x in s.then.stmts: self.gen_stmt(x)
            self.ind-=1; self.emit('}')
            if s.els:
                self.emit('else {'); self.ind+=1
                for x in s.els.stmts: self.gen_stmt(x)
                self.ind-=1; self.emit('}')
        elif isinstance(s,While):
            self.emit(f'while ({self.gen_expr(s.cond)}) {{'); self.ind+=1
            for x in s.body.stmts: self.gen_stmt(x)
            self.ind-=1; self.emit('}')
        elif isinstance(s,Return):
            if s.expr is None: self.emit('return;')
            else: self.emit(f'return {self.gen_expr(s.expr)};')
        elif isinstance(s,Print):
            first=True
            for a in s.args:
                if not first: self.emit('printf(" ");')
                first=False
                ty=self.expr_static_type(a); ce=self.gen_expr(a)
                if   ty==T_INT:   self.emit(f'__p_int({ce});')
                elif ty==T_DBL:   self.emit(f'__p_double({ce});')
                elif ty==T_BOOL:  self.emit(f'__p_bool({ce});')
                elif ty==T_STR:   self.emit(f'__p_string({ce});')
                else:              self.emit(f'__p_string("<?>");')
            self.emit('printf("\\n");')
        elif isinstance(s,ExprStmt):
            self.emit(self.gen_expr(s.expr) + ';')
    def expr_static_type(self,e):
        # Prefer analyzer annotation if present
        if hasattr(e,'ty') and e.ty: return e.ty
        if isinstance(e,Call): return self.ctx.get_fn(e.name).ret
        if isinstance(e,Unary):
            t=self.expr_static_type(e.rhs); return T_BOOL if e.op=='!' else t
        if isinstance(e,Binary):
            if e.op in ('*','/','+','-'): return T_DBL
            if e.op=='%': return T_INT
            return T_BOOL
        return None
    def gen_expr(self,e):
        if isinstance(e,Lit):
            if e.ty==T_INT: return str(int(e.val))
            if e.ty==T_DBL:
                s=repr(float(e.val))
                if '.' not in s and 'e' not in s and 'E' not in s: s+='.0'
                return s
            if e.ty==T_BOOL: return 'true' if e.val else 'false'
            if e.ty==T_STR:  return f'"{c_escape(e.val)}"'
        if isinstance(e,Var): return e.name
        if isinstance(e,Unary): return f'({e.op}{self.gen_expr(e.rhs)})'
        if isinstance(e,Binary): return f'({self.gen_expr(e.lhs)} {e.op} {self.gen_expr(e.rhs)})'
        if isinstance(e,Call):
            args=', '.join(self.gen_expr(a) for a in e.args)
            return f'{e.name}({args})'
        raise RuntimeError('unknown expr')

# ============================== Driver =============================
def compile_source(src:str)->str:
    toks=lex(src)
    ast=Parser(toks).parse()
    ctx=analyze(ast)
    return CGen(ast,ctx).gen()

def main():
    if len(sys.argv)!=2:
        print('Usage: echc.py <input.ech>  # outputs C on stdout', file=sys.stderr)
        sys.exit(1)
    src=open(sys.argv[1],'r',encoding='utf-8').read()
    try:
        sys.stdout.write(compile_source(src))
    except (SyntaxError,TypeError) as e:
        print(f'Compile error: {e}', file=sys.stderr)
        sys.exit(2)

if __name__=='__main__':
    main()

Run:

python3 echc.py hello.ech > hello.c
cc hello.c -o hello -lm
./hello

You said:
hello world in def function in ech
ChatGPT said:

Gotcha—here are two tiny Ech programs that say hello from a function (note: in Ech it’s fn, not def).

Option 1: function returns a string

fn hello() -> string {
  return "Hello, world!";
}

fn main() {
  print(hello());
}

Option 2: function prints directly

fn say_hello() -> void {
  print("Hello, world!");
}

fn main() {
  say_hello();
}

Compile/run:

python3 echc.py hello.ech > hello.c
cc hello.c -o hello -lm
./hello

You said:
(base) ian-martin-ajzenszmidt@ian-martin-ajzenszmidt-CFSZ5-3:~/Downloads$ python3 echc.py hello.ech > hello.c
cc hello.c -o hello -lm
./hello
// Warning: no main() defined; producing a library C file.
/usr/bin/ld: /usr/lib/gcc/x86_64-linux-gnu/13/../../../x86_64-linux-gnu/Scrt1.o: in function _start':
(.text+0x1b): undefined reference to main'
collect2: error: ld returned 1 exit status
bash: ./hello: No such file or directory
(base) ian-martin-ajzenszmidt@ian-martin-ajzenszmidt-CFSZ5-3:~/Downloads$ cc hello.c -o hello -lm
ChatGPT said:

You compiled a C file without a main. That warning is from the Ech compiler telling you your .ech source didn’t define fn main() { ... }.

Use this minimal hello.ech (exactly this):

fn main() {
  print("Hello, world!");
}

Then re-generate and run:

python3 echc.py hello.ech > hello.c
cc hello.c -o hello -lm
./hello

If you still see the “no main() defined” warning, double-check that:

    it’s fn main() (not def, not Main, not missing ()),

    the main function isn’t commented out with // or #.

If you intentionally want a library with no main, build it as a shared lib instead:

cc -shared -fPIC hello.c -o libhello.so -lm


ChatGPT can make mistakes. Check important info.
