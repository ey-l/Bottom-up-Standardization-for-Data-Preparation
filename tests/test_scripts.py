import ast
from typing import List, Union, Any, Tuple
import sys
import os
import astpretty
from itertools import combinations, permutations, product, chain
from collections import namedtuple, OrderedDict
import networkx as nx
import copy
import matplotlib.pyplot as plt
from pathlib import Path
import time

sys.path.append(sys.path[0] + '/..')
#print(sys.path)
from lucidscript.ASTDAG import *
from lucidscript.LUCIDDAG import *
from lucidscript.LUCID import *

FILES = []

def remove_empty_folders(path_abs):
    walk = list(os.walk(path_abs))
    for path, _, _ in walk[::-1]:
        if len(os.listdir(path)) == 0:
            os.remove(path)

if __name__ == "__main__":
    # Get all file paths
    verbose = True
    luciddags = []
    root = PROJ_ROOT / 'data' / 'lemmatized' / 'uciml_pima-indians-diabetes-database' #'house-prices-advanced-regression-techniques'
    pattern = "*.py"
    for path, subdirs, files in os.walk(root):
        for name in files:
            # Load scripts
            if name.split('.')[-1] == 'py' and name.split('.')[0] != "codex":
                with open(os.path.join(path, name), encoding='utf-8') as f:
                    code = f.read()
                    code = code.replace("data/input/", f"{sys.path[-1]}/data/input/")
                try:
                    tree = ast.parse(code)
                    d = ASTDAG(root=tree, filename=name)
                    d.gen_DAG()
                    luciddags.append(LUCIDDAG(d))
                    fp = os.path.join(path.split('/')[-1], name)
                    #if fp in FILES:
                    #    print(fp)
                    #    FILES.remove(fp)
                    print(fp)
                except Exception as e:
                    # ASTDAG.gen_DAG() returns NotImplemented
                    #print(e)
                    #to_remove = os.path.join(path, name)
                    #os.remove(os.path.join(root,to_remove))
                    #print(to_remove)
                    pass
    
    if verbose:
        print("Number of scripts:", len(luciddags))
    
    #remove_empty_folders(root)
    #luciddags

    # AtomCollection init
    #ac = AtomCollection(luciddags)
    #ac.report()

    # AtomCollection init
    #ac = AtomCollection(luciddags, tune=2)
    #ac.report()