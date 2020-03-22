import csv
#import pandas as pd
import json
from ast import literal_eval

#import findSurfaceResidues
from pymol import cmd

HEADERS_PROT = ["S", "ORF3a", "ORF8"]
HEADERS_MUT  = ["S Mutations", "orf3a Mutations", "orf8 Mutations"]

cmd.load("QHD43416.pdb", "S")
cmd.load("QHD43417.pdb", "ORF3a")
cmd.load("QHD43422.pdb", "ORF8")

content = {}

mat = {n:[] for n in HEADERS_MUT + HEADERS_PROT}

dr_content = []
with open("DataTable.tsv", "r") as csvf:
    df = csv.DictReader(csvf, delimiter='\t') #names=(HEADERS_PROT + HEADERS_MUT))
    for i, line in enumerate(df):
        op_line = line
        for j, kv_tup in enumerate(line):
            if kv_tup in HEADERS_MUT:
                #print(kv_tup)
                line[kv_tup] = literal_eval(line[kv_tup])
        dr_content.append(op_line)

selection_names = []

for i, line in enumerate(dr_content):
    #print(line)
    for j, key in enumerate(line):
        #print(line[kv_pair])
        if key not in (HEADERS_MUT):
            continue
        for k, mut in enumerate(line[key]):
            name = key.split()[0] + "_" +  str(i)
            cmd.select(name, "resi " + str(mut[1]) + " in " + key.split()[0])
            selection_names.append(name)

cmd.hide("all")
cmd.show("spheres")


#for prot in HEADERS_PROT:
#    cmd.alter(prot, "vdw=" + str(0.1))

for name in selection_names:
    cmd.alter(name, "cdw=" + str(2))
    cmd.color("red", name)