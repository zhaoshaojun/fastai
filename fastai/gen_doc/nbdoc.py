"`gen_doc.nbdoc` generates notebook documentation from module functions and links to correct places"

import inspect,importlib,enum,os,re
from IPython.core.display import display, Markdown, HTML
from typing import Dict, Any, AnyStr, List, Sequence, TypeVar, Tuple, Optional, Union
from .docstrings import *
from .core import *
from ..torch_core import *
__all__ = ['get_fn_link', 'link_docstring', 'show_doc', 'get_ft_names',
           'get_exports', 'show_video', 'show_video_from_youtube', 'create_anchor', 'import_mod', 'get_source_link', 'is_enum']

MODULE_NAME = 'fastai'
SOURCE_URL = 'https://github.com/fastai/fastai_pytorch/blob/master/'
PYTORCH_DOCS = 'https://pytorch.org/docs/stable/'
_typing_names = {t:n for t,n in fastai_types.items() if t.__module__=='typing'}

def is_enum(cls): return cls == enum.Enum or cls == enum.EnumMeta

def link_type(arg_type, arg_name=None, include_bt:bool=True):
    "creates link to documentation"
    arg_name = arg_name or fn_name(arg_type)
    if include_bt: arg_name = code_esc(arg_name)
    if is_fastai_class(arg_type): return f'[{arg_name}]({get_fn_link(arg_type)})'
    if belongs_to_module(arg_type, 'torch') and ('Tensor' not in arg_name): return f'[{arg_name}]({get_pytorch_link(arg_type)})'
    return arg_name

def is_fastai_class(t): return belongs_to_module(t, MODULE_NAME)

def belongs_to_module(t, module_name):
    "checks if belongs to module_name"
    if not inspect.getmodule(t): return False
    return inspect.getmodule(t).__name__.startswith(module_name)

def code_esc(s): return f'`{s}`'

def type_repr(t):
    if t in _typing_names: return link_type(t, _typing_names[t])
    if hasattr(t, '__forward_arg__'): return link_type(t.__forward_arg__)
    elif getattr(t, '__args__', None):
        args = t.__args__
        if len(args)==2 and args[1] == type(None):
            return f'`Optional`\[{type_repr(args[0])}\]'
        reprs = ', '.join([type_repr(o) for o in t.__args__])
        return f'{link_type(t)}\[{reprs}\]'
    else: return link_type(t)

def anno_repr(a): return type_repr(a)

def format_param(p):
    res = code_esc(p.name)
    if hasattr(p, 'annotation') and p.annotation != p.empty: res += f':{anno_repr(p.annotation)}'
    if p.default != p.empty:
        default = getattr(p.default, 'func', p.default)
        default = getattr(default, '__name__', default)
        res += f'=`{repr(default)}`'
    return res

def format_ft_def(func, full_name:str=None)->str:
    "Formats and links function definition to show in documentation"
    sig = inspect.signature(func)
    name = f'`{ifnone(full_name, func.__name__)}`'
    fmt_params = [format_param(param) for name,param
                  in sig.parameters.items() if name not in ('self','cls')]
    arg_str = f"({', '.join(fmt_params)})"
    if sig.return_annotation and (sig.return_annotation != sig.empty): arg_str += f" -> {anno_repr(sig.return_annotation)}"
    if is_fastai_class(type(func)):        arg_str += f" :: {link_type(type(func))}"
    f_name = f"`class` {name}" if inspect.isclass(func) else name
    return f'{f_name}\n> {name}{arg_str}'

def get_enum_doc(elt, full_name:str) -> str:
    "Formatted enum documentation"
    vals = ', '.join(elt.__members__.keys())
    doc = f'{code_esc(full_name)}\n`Enum` = [{vals}]'
    return doc

def get_cls_doc(elt, full_name:str) -> str:
    "Class definition"
    parent_class = inspect.getclasstree([elt])[-1][0][1][0]
    doc = format_ft_def(elt, full_name)
    if parent_class != object: doc += f' :: {link_type(parent_class, include_bt=True)}'
    return doc

def show_doc(elt, doc_string:bool=True, full_name:str=None, arg_comments:dict=None, title_level=None, alt_doc_string:str='',
             ignore_warn:bool=False, markdown=True):
    "Show documentation for element `elt`. Supported types: class, Callable, and enum"
    arg_comments = ifnone(arg_comments, {})
    if full_name is None and hasattr(elt, '__name__'): full_name = elt.__name__
    if inspect.isclass(elt):
        if is_enum(elt.__class__):   doc = get_enum_doc(elt, full_name)
        else:                        doc = get_cls_doc(elt, full_name)
    elif isinstance(elt, Callable):  doc = format_ft_def(elt, full_name)
    else: doc = f'doc definition not supported for {full_name}'
    title_level = ifnone(title_level, 2 if inspect.isclass(elt) else 4)
    link = f'<a id={full_name}></a>'
    doc += '\n'
    if doc_string and (inspect.getdoc(elt) or arg_comments):
        doc += format_docstring(elt, arg_comments, alt_doc_string, ignore_warn) + ' '
    if is_fastai_class(elt): doc += get_function_source(elt)
    # return link+doc
    display(title_md(link+doc, title_level, markdown=markdown))

def format_docstring(elt, arg_comments:dict={}, alt_doc_string:str='', ignore_warn:bool=False) -> str:
    "merges and formats the docstring definition with arg_comments and alt_doc_string"
    parsed = ""
    doc = parse_docstring(inspect.getdoc(elt))
    description = alt_doc_string or doc['long_description'] or doc['short_description']
    if description: parsed += f'\n\n{link_docstring(inspect.getmodule(elt), description)}'

    resolved_comments = {**doc.get('comments', {}), **arg_comments} # arg_comments takes priority
    args = inspect.getfullargspec(elt).args if not is_enum(elt.__class__) else elt.__members__.keys()
    if resolved_comments: parsed += '\n'
    for a in resolved_comments:
        parsed += f'\n- *{a}*: {resolved_comments[a]}'
        if a not in args and not ignore_warn: warn(f'Doc arg mismatch: {a}')

    return_comment = arg_comments.get('return') or doc.get('return')
    if return_comment: parsed += f'\n\n*return*: {return_comment}'
    return parsed

# Finds all places with a backtick but only if it hasn't already been linked
BT_REGEX = re.compile("\[`([^`]*)`\](?:\([^)]*\))|`([^`]*)`") # matches [`key`](link) or `key`
def link_docstring(modules, docstring:str, overwrite:bool=False) -> str:
    "searches `docstring` for backticks and attempts to link those functions to respective documentation"
    mods = listify(modules)
    modvars = {}
    for mod in mods: modvars.update(mod.__dict__) # concat all module definitions
    for m in BT_REGEX.finditer(docstring):
        keyword = m.group(1) or m.group(2)
        elt = find_elt(modvars, keyword)
        if elt is None: continue
        link = link_type(elt, arg_name=keyword)
        docstring = docstring.replace(m.group(0), link) # group(0) = replace whole link with new one
    return docstring

def find_elt(modvars, keyword, match_last=True):
    "Attempts to resolve keywords such as Learner.lr_find. `match_last` starts matching from last component."
    if keyword in modvars: return modvars[keyword]
    if '.' not in keyword: return None
    comps = keyword.split('.')
    if match_last: return modvars.get(comps[-1])
    comp_elt = modvars.get(comps[0])
    if hasattr(comp_elt, '__dict__'):
        return find_elt(comp_elt.__dict__, '.'.join(comps[1:]))

def import_mod(mod_name:str):
    "returns module from `mod_name`"
    splits = str.split(mod_name, '.')
    try:
        if len(splits) > 1 : mod = importlib.import_module('.' + '.'.join(splits[1:]), splits[0])
        else: mod = importlib.import_module(mod_name)
        return mod
    except: print(f"Module {mod_name} doesn't exist.")

def show_doc_from_name(mod_name, ft_name:str, doc_string:bool=True, arg_comments:dict={}, alt_doc_string:str=''):
    "shows documentation for `ft_name`. see `show_doc`"
    mod = import_mod(mod_name)
    splits = str.split(ft_name, '.')
    assert hasattr(mod, splits[0]), print(f"Module {mod_name} doesn't have a function named {splits[0]}.")
    elt = getattr(mod, splits[0])
    for i,split in enumerate(splits[1:]):
        assert hasattr(elt, split), print(f"Class {'.'.join(splits[:i+1])} doesn't have a function named {split}.")
        elt = getattr(elt, split)
    show_doc(elt, doc_string, ft_name, arg_comments, alt_doc_string)

def get_exports(mod):
    public_names = mod.__all__ if hasattr(mod, '__all__') else dir(mod)
    #public_names.sort(key=str.lower)
    return [o for o in public_names if not o.startswith('_')]

def get_ft_names(mod, include_inner=False)->List[str]:
    "Returns all the functions of module `mod`"
    # If the module has an attribute __all__, it picks those.
    # Otherwise, it returns all the functions defined inside a module.
    fn_names = []
    for elt_name in get_exports(mod):
        elt = getattr(mod,elt_name)
        #This removes the files imported from elsewhere
        try:    fname = inspect.getfile(elt)
        except: continue
        if mod.__file__.endswith('__init__.py'):
            if inspect.ismodule(elt): fn_names.append(elt_name)
            else: continue
        else:
            if (fname != mod.__file__): continue
            if inspect.isclass(elt) or inspect.isfunction(elt): fn_names.append(elt_name)
            else: continue
        if include_inner and inspect.isclass(elt) and not is_enum(elt.__class__):
            fn_names.extend(get_inner_fts(elt))
    return fn_names

def get_inner_fts(elt) -> List[str]:
    "return methods belonging to class"
    fts = []
    for ft_name in elt.__dict__.keys():
        if ft_name.startswith('_'): continue
        ft = getattr(elt, ft_name)
        if inspect.isfunction(ft): fts.append(f'{elt.__name__}.{ft_name}')
        if inspect.isclass(ft): fts += [f'{elt.__name__}.{n}' for n in get_inner_fts(ft)]
    return fts

def get_module_toc(mod_name):
    "displays table of contents for given `mod_name`"
    mod = import_mod(mod_name)
    ft_names = mod.__all__ if hasattr(mod,'__all__') else get_ft_names(mod)
    ft_names.sort(key = str.lower)
    tabmat = ''
    for ft_name in ft_names:
        tabmat += f'- [{ft_name}](#{ft_name})\n'
        elt = getattr(mod, ft_name)
        if inspect.isclass(elt) and not is_enum(elt.__class__):
            in_ft_names = get_inner_fts(elt)
            for name in in_ft_names:
                tabmat += f'  - [{name}](#{name})\n'
    display(Markdown(tabmat))

def get_class_toc(mod_name:str, cls_name:str):
    "displays table of contents for `cls_name`"
    splits = str.split(mod_name, '.')
    try: mod = importlib.import_module('.' + '.'.join(splits[1:]), splits[0])
    except:
        print(f"Module {mod_name} doesn't exist.")
        return
    splits = str.split(cls_name, '.')
    assert hasattr(mod, splits[0]), print(f"Module {mod_name} doesn't have a function named {splits[0]}.")
    elt = getattr(mod, splits[0])
    for i,split in enumerate(splits[1:]):
        assert hasattr(elt, split), print(f"Class {'.'.join(splits[:i+1])} doesn't have a subclass named {split}.")
        elt = getattr(elt, split)
    assert inspect.isclass(elt) and not is_enum(elt.__class__), "This is not a valid class."
    in_ft_names = get_inner_fts(elt)
    tabmat = ''
    for name in in_ft_names: tabmat += f'- [{name}](#{name})\n'
    display(Markdown(tabmat))

def show_video(url):
    data = f'<iframe width="560" height="315" src="{url}" frameborder="0" allowfullscreen></iframe>'
    return display(HTML(data))

def show_video_from_youtube(code, start=0):
    url = f'https://www.youtube.com/embed/{code}?start={start}&amp;rel=0&amp;controls=0&amp;showinfo=0'
    return show_video(url)

def fn_name(ft)->str:
    if ft in _typing_names: return _typing_names[ft]
    if hasattr(ft, '__name__'):   return ft.__name__
    elif hasattr(ft,'_name') and ft._name: return ft._name
    #elif hasattr(ft,'__class__'): return ft.__class__.__name__
    elif hasattr(ft,'__origin__'): return str(ft.__origin__).split('.')[-1]
    else:                         return str(ft).split('.')[-1]

def get_fn_link(ft) -> str:
    "returns function link to notebook documentation"
    strip_name = strip_fastai(get_module_name(ft))
    func_name = strip_fastai(fn_name(ft))
    return f'/{strip_name}.html#{func_name}'

def get_module_name(ft) -> str: return ft.__name__ if inspect.ismodule(ft) else ft.__module__

def get_pytorch_link(ft) -> str:
    "returns link to pytorch docs"
    name = ft.__name__
    if name.startswith('torch.nn') and inspect.ismodule(ft): # nn.functional is special case
        nn_link = name.replace('.', '-')
        return f'{PYTORCH_DOCS}nn.html#{nn_link}'
    paths = get_module_name(ft).split('.')
    if len(paths) == 1: return f'{PYTORCH_DOCS}{paths[0]}.html#{paths[0]}.{name}'

    offset = 1 if paths[1] == 'utils' else 0 # utils is a pytorch special case
    doc_path = paths[1+offset]
    fnlink = '.'.join(paths[:(2+offset)]+[name])
    return f'{PYTORCH_DOCS}{doc_path}.html#{fnlink}'


def get_source_link(mod, lineno) -> str:
    "returns link to  line in source code"
    github_path = mod.__name__.replace('.', '/')
    link = f"{SOURCE_URL}{github_path}.py#L{lineno}"
    return f'<a href="{link}">[source]</a>'

def get_function_source(ft) -> str:
    "returns link to  line in source code"
    lineno = inspect.getsourcelines(ft)[1]
    return get_source_link(inspect.getmodule(ft), lineno)

def title_md(s:str, title_level:int, markdown=True):
    res = '#' * title_level
    if title_level: res += ' '
    return Markdown(res+s) if markdown else (res+s)

def create_anchor(text, title_level=0, name=None):
    if name is None: name=str2id(text)
    display(title_md(f'<a id={name}></a>{text}'))

