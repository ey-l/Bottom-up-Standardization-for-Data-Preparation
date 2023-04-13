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

def load_luciddag_from_path(path):
    with open(path, encoding='utf-8') as f:
        code = f.read()
        code = code.replace("_/Users", "/Users")
        code = code.replace("_data", "data")
        #code = code.replace("data/input/", f"{sys.path[-1]}/data/input/")
    tree = ast.parse(code)
    d = ASTDAG(root=tree, filename=name)
    try:
        d.gen_DAG()
        return LUCIDDAG(d)
    except Exception as e:
        return None

if __name__ == "__main__":
    # Get all file paths
    method = "codex"
    dataset = 'competitive-data-science-predict-future-sales' #'house-prices-advanced-regression-techniques'
    verbose = True
    results = {}
    corpus_sets = {}
    scripts = {}
    root = PROJ_ROOT / 'scratch' / method / dataset
    pattern = "*.py"
    scriptpath = ""
    sdirs = []
    for path, subdirs, files in os.walk(root):
        if len(subdirs) != 0: sdirs = subdirs
        for subdir in subdirs:
            filepath = os.path.join(os.path.join(path, subdir), method+".py")
            if os.path.exists(filepath):
                #print(filepath)
                try:
                    ld = load_luciddag_from_path(filepath)
                    if ld is not None:
                        results[subdir] = ld
                except Exception as e:
                    print("Couldn't parse:",filepath)

    for subdir in sdirs:
        for path, subdirs, files in os.walk(os.path.join(root, subdir)):
            for name in files:
                if name.split('.')[-1] == 'py' and name.split('.')[0] != method:
                    scriptpath = os.path.join(os.path.join(root, subdir), name)
                    ld = load_luciddag_from_path(scriptpath)
                    if ld is not None:
                        scripts[subdir] = ld
            
    # Load corpus set
    for subdir in sdirs:
        corpus_sets[subdir] = []
        for script in scripts:
            if script != subdir:
                corpus_sets[subdir].append(scripts[script])

    
    if verbose:
        print(len(scripts))
        print(len(results))
        print("Number of scripts:", len(corpus_sets[sdirs[0]]))

    output_dir = PROJ_ROOT / 'exp' / method / dataset
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Record time
    filepath = output_dir / f'stats.csv'
    columns = ["function", "time", "var", "val", "script_i"]
    log_dict = {}
    log_dict["filepath"] = filepath
    log_dict["header"] = LOG_HEADER + columns

    # AtomCollection init
    #ac = AtomCollection(list(scripts.values())[:5])
    ac = AtomCollection(scripts.values())
    ac.report()
    i = 0
    data = {}

    for subdir in results:

        # Initialize Px after building AtomCollection
        #for luciddag in scripts:
        print("====================")
        print(subdir)
        data[subdir] = {}
        result = results[subdir]
        result.init_Px(ac)

        script = scripts[subdir]
        script.init_Px(ac)

        lucid = Lucid(output_dir=output_dir, AC=ac, scripts=corpus_sets[subdir])
        data[subdir]["script_i"] = i
        data[subdir]["script"] = lucid.compute_quality(script.Px)
        data[subdir]["method"] = lucid.compute_quality(result.Px)

        lucid.set_input(script)
        lucid.set_script_i(i)
        correctness_filepath = lucid.init_data_correctness_eval()
        print(correctness_filepath)
        if len(correctness_filepath) == 0:
            print("==========================")
            i += 1
            continue
        
        # Run correctness eval
        try:
            lucid.eval_data_correctness(i, result, 0)
            print("SUCCESS")
        except Exception as ex:
            print(ex)
        i += 1
    
    df = pd.DataFrame.from_dict(data, orient='index')
    df.reset_index(inplace=True)
    df = df.rename(columns = {'index':'subdir'})
    df.to_csv(os.path.join(str(output_dir), "results.csv"), index=False)
    print(df.head())
    # Setups