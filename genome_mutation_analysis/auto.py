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
            if kv_tup[0] in HEADERS_MUT:
                op_line[1] = literal_eval(kv_tup[1])
        dr_content.append(op_line)

#list list key value pair

#print(dr_content[0])



for line in dr_content:
    for kv_pair in line:
        if kv_pair[0] not in (HEADERS_MUT):
            continue
        for mut in kv_pair[1]:
            cmd.select("resi " + str(mut[1]))

#cmd.show("cartoon", "sele")

