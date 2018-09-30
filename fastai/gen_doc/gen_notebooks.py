"`gen_doc.nbdoc` generates notebook documentation from module functions and links to correct places"
import pkgutil, inspect, sys,os, importlib,json,enum,warnings,nbformat,re
from IPython.core.display import display, Markdown
from nbconvert.preprocessors import ExecutePreprocessor
from nbformat.sign import NotebookNotary
from pathlib import Path
from .core import *
from .nbdoc import *

__all__ = ['create_module_page', 'generate_all', 'update_module_page', 'update_all', 'link_all', 'import_mod',
           'link_nb', 'update_notebooks']

def get_empty_notebook():
    "a default notbook with the minimum metadata"
    #TODO: check python version and nbformat
    return {'metadata': {'kernelspec': {'display_name': 'Python 3',
                                        'language': 'python',
                                        'name': 'python3'},
                         'language_info': {'codemirror_mode': {'name': 'ipython', 'version': 3},
                         'file_extension': '.py',
                         'mimetype': 'text/x-python',
                         'name': 'python',
                         'nbconvert_exporter': 'python',
                         'pygments_lexer': 'ipython3',
                         'version': '3.6.6'}},
            'nbformat': 4,
            'nbformat_minor': 2}

def get_md_cell(source, metadata=None):
    "a markdown cell containing the source text"
    return {'cell_type': 'markdown',
            'metadata': {} if metadata is None else metadata,
            'source': source}

def get_empty_cell(ctype='markdown'):
    "an empty cell of type ctype"
    return {'cell_type': ctype, 'metadata': {}, 'source': []}

def get_code_cell(code, hidden=False):
    "a code cell containing the code"
    return {'cell_type' : 'code',
            'execution_count': 0,
            'metadata' : {'hide_input': hidden, 'trusted':True},
            'source' : code,
            'outputs': []}

def get_doc_cell(ft_name):
    "a code cell with the command to show the doc of a given function"
    code = f"show_doc({ft_name})"
    return get_code_cell(code, True)

def get_inner_fts(elt):
    "List the inner functions of a class"
    fts = []
    for ft_name in elt.__dict__.keys():
        if ft_name[:2] == '__': continue
        ft = getattr(elt, ft_name)
        if inspect.isfunction(ft): fts.append(f'{elt.__name__}.{ft_name}')
        if inspect.ismethod(ft): fts.append(f'{elt.__name__}.{ft_name}')
        if inspect.isclass(ft): fts += [f'{elt.__name__}.{n}' for n in get_inner_fts(ft)]
    return fts

def get_global_vars(mod):
    "Returns globally assigned variables"
    # https://stackoverflow.com/questions/8820276/docstring-for-variable/31764368#31764368
    import ast,re
    with open(mod.__file__, 'r') as f: fstr = f.read()
    flines = fstr.splitlines()
    d = {}
    for node in ast.walk(ast.parse(fstr)):
        if isinstance(node,ast.Assign) and hasattr(node.targets[0], 'id'):
            key,lineno = node.targets[0].id,node.targets[0].lineno
            codestr = flines[lineno]
            match = re.match(f"^({key})\s*=\s*.*", codestr)
            if match and match.group(1) != '__all__': # only top level assignment
                d[key] = f'`{codestr}` {get_source_link(mod, lineno)}'
    return d

def execute_nb(fname):
    "Execute notebook `fname`"
    # Any module used in the notebook that isn't inside must be in the same directory as this script

    with open(fname) as f: nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    ep.preprocess(nb, {})
    with open(fname, 'wt') as f: nbformat.write(nb, f)
    NotebookNotary().sign(nb)

def _symbol_skeleton(name): return [get_doc_cell(name), get_md_cell(f"`{name}`")]

def create_module_page(mod, dest_path, force=False):
    "Creates the documentation notebook for module `mod_name` in path `dest_path`"
    nb = get_empty_notebook()
    mod_name = mod.__name__
    strip_name = strip_fastai(mod_name)
    init_cell = [get_md_cell(f'# {strip_name}'), get_md_cell('Type an introduction of the package here.')]
    add_module_metadata(mod, init_cell)
    cells = [get_code_cell(f'from fastai.gen_doc.nbdoc import *\nfrom {mod_name} import * ', True)]

    gvar_map = get_global_vars(mod)
    if gvar_map: cells.append(get_md_cell('### Global Variable Definitions:'))
    for name in get_exports(mod):
        if name in gvar_map: cells.append(get_md_cell(gvar_map[name]))

    for ft_name in get_ft_names(mod, include_inner=True):
        if not hasattr(mod, ft_name):
            warnings.warn(f"Module {strip_name} doesn't have a function named {ft_name}.")
            continue
        cells += _symbol_skeleton(ft_name)
        elt = getattr(mod, ft_name)
    nb['cells'] = init_cell + cells + [get_md_cell(UNDOC_HEADER)]

    doc_path = get_doc_path(mod, dest_path)
    json.dump(nb, open(doc_path, 'w' if force else 'x'))
    execute_nb(doc_path)
    return doc_path

_default_exclude = ['.ipynb_checkpoints', '__pycache__', '__init__.py', 'imports']

def get_module_names(path_dir, exclude=None):
    if exclude is None: exclude = _default_exclude
    "Searches a given directory and returns all the modules contained inside"
    files = path_dir.glob('*')
    res = []
    for f in files:
        if f.is_dir() and f.name in exclude: continue # exclude directories
        if any([f.name.endswith(ex) for ex in exclude]): continue # exclude extensions

        if f.name[-3:] == '.py': res.append(f'{path_dir.name}.{f.name[:-3]}')
        elif f.is_dir(): res += [f'{path_dir.name}.{name}' for name in get_module_names(f)]
    return res

def generate_all(pkg_name, dest_path, exclude=None):
    "Generate the documentation for all the modules in `pkg_name`"
    if exclude is None: exclude = _default_exclude
    mod_files = get_module_names(Path(pkg_name), exclude)
    for mod_name in mod_files:
        mod = import_mod(mod_name)
        if mod is None: continue
        create_module_page(mod, dest_path)

def read_nb(fname):
    "Read a notebook and returns its corresponding json"
    with open(fname,'r') as f: return nbformat.reads(f.read(), as_version=4)

def read_nb_content(cells, mod_name):
    "Builds a dictionary containing the position of the cells giving the document for functions in a notebook"
    doc_fns = {}
    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'code':
            for match in re.findall(r"show_doc\(([\w\.]*)", cell['source']):
                doc_fns[match] = i
    return doc_fns

def read_nb_types(cells):
    doc_fns = {}
    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'markdown':
            match = re.match(r"^(?:<code>|`)?(\w*)\s*=\s*", cell['source'])
            if match is not None: doc_fns[match.group(1)] = i
    return doc_fns

def link_markdown_cells(cells, modules):
    "Creates documentation links for all cells in markdown with backticks"
    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'markdown':
            cell['source'] = link_docstring(modules, cell['source'])

def get_insert_idx(pos_dict, name):
    "Return the position to insert a given function doc in a notebook"
    keys,i = list(pos_dict.keys()),0
    while i < len(keys) and str.lower(keys[i]) < str.lower(name): i+=1
    if i == len(keys): return -1
    else:              return pos_dict[keys[i]]

def update_pos(pos_dict, start_key, nbr=2):
    "Updates the position dictionary by moving all positions after start_ket by nbr"
    for key,idx in pos_dict.items():
        if str.lower(key) >= str.lower(start_key): pos_dict[key] += nbr
    return pos_dict

def insert_cells(cells, pos_dict, ft_name, append=False):
    "Insert the function doc cells of a function in the list of cells at their correct position and updates the position dictionary"
    idx = get_insert_idx(pos_dict, ft_name)
    if append or idx == -1: cells += [get_doc_cell(ft_name), get_empty_cell()]
    else:
        cells.insert(idx, get_doc_cell(ft_name))
        cells.insert(idx+1, get_empty_cell())
        pos_dict = update_pos(pos_dict, ft_name, 2)
    return cells, pos_dict

def get_doc_path(mod, dest_path):
    strip_name = strip_fastai(mod.__name__)
    return os.path.join(dest_path,f'{strip_name}.ipynb')

def add_module_metadata(mod, cells):
    if has_metadata_cell(cells): return
    mcode = (f'from fastai.gen_doc.gen_notebooks import update_module_metadata\n'
             f'import {mod.__name__}\n'
             f'# For updating jekyll metadata. You MUST reload notebook immediately after executing this cell for changes to save\n'
             f'# Leave blank to autopopulate from mod.__doc__\n'
             f'update_module_metadata({mod.__name__})')
    cells.insert(0, get_code_cell(mcode, hidden=True))

def update_module_metadata(mod, dest_path='.', title=None, summary=None, keywords=None, overwrite=True):
    "Creates jekyll metadata for given module. Title and summary are autopoulated (if None) from module.__name__ and module.__doc__."
    title = title or strip_fastai(mod.__name__)
    summary = summary or inspect.getdoc(mod)
    update_nb_metadata(get_doc_path(mod, dest_path), title, summary, keywords, overwrite)

def update_nb_metadata(nb_path=None, title=None, summary=None, keywords=None, overwrite=True):
    "Creates jekyll metadata for given notebook path"
    nb = read_nb(nb_path)
    jm = {'title': title, 'summary': summary, 'keywords': keywords}
    update_metadata(nb, jm, overwrite)
    json.dump(nb, open(nb_path, 'w'))

METADATA_RE = re.compile(r"update_\w+_metadata")
def has_metadata_cell(cells):
    for c in cells: 
        if c['cell_type'] == 'code' and METADATA_RE.search(c['source']): return c

def add_nb_metadata(nb, nb_path):
    cells = nb['cells']
    if has_metadata_cell(cells): return
    jmb = nb['metadata'].get('jekyll', {})
    title, summary = stringify(jmb.get('title')), stringify(jmb.get('summary'))
    mcode = (f"from fastai.gen_doc.gen_notebooks import update_nb_metadata\n"
             f"# For updating jekyll metadata. You MUST reload notebook immediately after executing this cell for changes to save\n"
             f"update_nb_metadata('{Path(nb_path).name}', title={title}, summary={summary})")
    metadata_cell = get_code_cell(mcode, hidden=True)
    cells.insert(0, metadata_cell)

def stringify(s): return f'\'{s}\'' if isinstance(s, str) else s

def update_metadata(nb, data, overwrite=True):
    "Creates jekyll metadata. Always overwrites existing"
    data = {k:v for (k,v) in data.items() if v is not None} # remove none values
    if not data: return
    if 'metadata' not in nb: nb['metadata'] = {}
    if overwrite: nb['metadata']['jekyll'] = data
    else: nb['metadata']['jekyll'] = nb['metadata'].get('jekyll', {}).update(data)

IMPORT_RE = re.compile(r"from (fastai[\.\w_]*)")
def get_imported_modules(cells):
    module_names = ['fastai']
    for cell in cells:
        if cell['cell_type'] == 'code':
            for m in IMPORT_RE.finditer(cell['source']):
                if m.group(1) not in module_names: module_names.append(m.group(1))
    mods = [import_mod(m) for m in module_names]
    return [m for m in mods if m is not None]

NEW_FT_HEADER = '## New Methods - Please document or move to the undocumented section'
UNDOC_HEADER = '## Undocumented Methods - Methods moved below this line will intentionally be hidden'
def parse_sections(cells):
    old_cells, undoc_cells, new_cells = [], [], []
    current_section = old_cells
    for cell in cells:
        if cell['cell_type'] == 'markdown':
            if re.match(UNDOC_HEADER, cell['source']): current_section = undoc_cells
            if re.match(NEW_FT_HEADER, cell['source']): current_section = new_cells
        current_section.append(cell)
    undoc_cells = undoc_cells or [get_md_cell(UNDOC_HEADER)]
    new_cells = new_cells or [get_md_cell(NEW_FT_HEADER)]
    return old_cells, undoc_cells, new_cells

def remove_undoc_cells(cells):
    old, _, _ = parse_sections(cells)
    return old

def update_module_page(mod, dest_path='.'):
    "Updates the documentation notebook of a given module"
    doc_path = get_doc_path(mod, dest_path)
    strip_name = strip_fastai(mod.__name__)
    nb = read_nb(doc_path)
    cells = nb['cells']
    
    add_module_metadata(mod, cells)
    link_markdown_cells(cells, get_imported_modules(cells))

    type_dict = read_nb_types(cells)
    gvar_map = get_global_vars(mod)
    for name in get_exports(mod):
        if name not in gvar_map: continue
        code = gvar_map[name]
        if name in type_dict: cells[type_dict[name]] = get_md_cell(code)
        else: cells.append(get_md_cell(code))

    pos_dict = read_nb_content(cells, strip_name)
    ft_names = get_ft_names(mod, include_inner=True)
    new_fts = list(set(ft_names) - set(pos_dict.keys()))
    if new_fts: print(f'Found new fuctions for {mod}. Please document:\n{new_fts}')
    existing, undoc_cells, new_cells = parse_sections(cells)
    for ft_name in new_fts: new_cells.extend([get_doc_cell(ft_name), get_empty_cell()])
    if len(new_cells) > 1: nb['cells'] = existing + undoc_cells + new_cells

    json.dump(nb, open(doc_path,'w'))
    return doc_path
    #execute_nb(doc_path)

def link_nb(nb_path):
    nb = read_nb(nb_path)
    cells = nb['cells']
    link_markdown_cells(cells, get_imported_modules(cells))
    add_nb_metadata(nb, nb_path)
    json.dump(nb, open(nb_path,'w'))
    NotebookNotary().sign(read_nb(nb_path))

def link_all(path_dir):
    "Links documentation to all the notebooks in `pkg_name`"
    files = Path(path_dir).glob('*.ipynb')
    for f in files: link_nb(f)

def update_all(pkg_name, dest_path='.', exclude=None, create_missing=False):
    "Updates all the notebooks in `pkg_name`"
    if exclude is None: exclude = _default_exclude
    mod_files = get_module_names(Path(pkg_name), exclude)
    for f in mod_files:
        mod = import_mod(f)
        if mod is None: continue
        if os.path.exists(get_doc_path(mod, dest_path)):
            update_module_page(mod, dest_path)
        elif create_missing:
            print(f'Creating module page of {f}')
            create_module_page(mod, dest_path)

def resolve_path(path):
    "Creates absolute path if relative is provided"
    p = Path(path)
    if p.is_absolute(): return p
    return Path.cwd()/path

def get_module_from_path(source_path):
    "Finds module given a source path. Assumes it belongs to fastai directory"
    fpath = Path(source_path).resolve()
    if 'fastai' not in fpath.parts: 
        print(f'Could not resolve file {fpath}. source_path must be inside `fastai` directory', fpath)
        return []
    fastai_idx = list(reversed(fpath.parts)).index('fastai')
    dirpath = fpath.parents[fastai_idx]
    relpath = fpath.relative_to(dirpath)
    return '.'.join(relpath.with_suffix('').parts)

def update_notebooks(source_path, dest_path=None, do_all=False, update_html=True, update_nb=False,
                     update_nb_links=True, html_path=None, create_missing_docs=False):
    "`source_path` can be a directory or a file. Assumes all modules reside in the fastai directory."
    from .convert2html import convert_all, convert_nb
    fpath = Path(__file__).resolve()
    fastai_idx = list(reversed(fpath.parts)).index('fastai')
    dirpath = fpath.parents[fastai_idx] # should return 'fastai_pytorch'
    source_path = resolve_path(source_path)
    if dest_path is None: dest_path = dirpath/'docs_src'
    else: dest_path = resolve_path(dest_path)
    if html_path is None: html_path = dirpath/'docs'
    else: html_path = resolve_path(html_path)

    if source_path.is_file():
        doc_path = source_path
        if update_nb and (source_path.suffix == '.py'):
            mod = import_mod(get_module_from_path(source_path))
            if not mod: return print('Could not find module for path:', source_path)
            try: doc_path = update_module_page(mod, dest_path)
            except FileNotFoundError:
                if create_missing_docs: doc_path = create_module_page(mod, dest_path)
                else: print(f'Could not update file {source_path}. Please set create_missing_docs=True')
        if update_nb_links: link_nb(doc_path)
        if update_html: convert_nb(doc_path, html_path)
    elif source_path.is_dir():
        if update_nb: update_all(source_path, dest_path, create_missing=create_missing_docs)
        if update_nb_links: link_all(dest_path)
        if update_html: convert_all(dest_path, html_path)
    else: print('Could not resolve source file:', source_path)

