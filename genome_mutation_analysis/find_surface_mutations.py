import csv
import json
from ast import literal_eval

from pymol import cmd




###################

'''
http://pymolwiki.org/index.php/FindSurfaceResidues

Copyright <?> <Jason Vertrees>

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

def findSurfaceAtoms(selection="all", cutoff=2.5, quiet=1):
    """
DESCRIPTION

    Finds those atoms on the surface of a protein
    that have at least 'cutoff' exposed A**2 surface area.

USAGE

    findSurfaceAtoms [ selection, [ cutoff ]]

SEE ALSO

    findSurfaceResidues
    """
    cutoff, quiet = float(cutoff), int(quiet)

    tmpObj = cmd.get_unused_name("_tmp")
    cmd.create(tmpObj, "(" + selection + ") and polymer", zoom=0)

    cmd.set("dot_solvent", 1, tmpObj)
    cmd.get_area(selection=tmpObj, load_b=1)

    # threshold on what one considers an "exposed" atom (in A**2):
    cmd.remove(tmpObj + " and b < " + str(cutoff))

    selName = cmd.get_unused_name("exposed_atm_")
    cmd.select(selName, "(" + selection + ") in " + tmpObj)

    cmd.delete(tmpObj)

    if not quiet:
        print("Exposed atoms are selected in: " + selName)

    return selName


def findSurfaceResidues(selection="all", cutoff=2.5, doShow=0, quiet=1):
    """
DESCRIPTION

    Finds those residues on the surface of a protein
    that have at least 'cutoff' exposed A**2 surface area.

USAGE

    findSurfaceResidues [ selection, [ cutoff, [ doShow ]]]

ARGUMENTS

    selection = string: object or selection in which to find exposed
    residues {default: all}

    cutoff = float: cutoff of what is exposed or not {default: 2.5 Ang**2}

RETURNS

    (list: (chain, resv ) )
        A Python list of residue numbers corresponding
        to those residues w/more exposure than the cutoff.

    """
    cutoff, doShow, quiet = float(cutoff), int(doShow), int(quiet)

    selName = findSurfaceAtoms(selection, cutoff, quiet)

    exposed = set()
    cmd.iterate(selName, "exposed.add((chain,resv))", space=locals())

    selNameRes = cmd.get_unused_name("exposed_res_")
    cmd.select(selNameRes, "byres " + selName)

    if not quiet:
        print("Exposed residues are selected in: " + selNameRes)

    if doShow:
        cmd.show_as("spheres", "(" + selection + ") and polymer")
        cmd.color("white", selection)
        cmd.color("yellow", selNameRes)
        cmd.color("red", selName)

    return sorted(exposed)

cmd.extend("findSurfaceAtoms", findSurfaceAtoms)
cmd.extend("findSurfaceResidues", findSurfaceResidues)


#######################



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
mutations = []

for i, line in enumerate(dr_content):
    #print(line)
    for j, key in enumerate(line):
        #print(line[kv_pair])
        if key not in (HEADERS_MUT):
            continue
        for k, mut in enumerate(line[key]):
            name = key.split()[0] + "_" +  str(i)
            cmd.select(name, "resi " + str(mut[1]) + " in " + key.split()[0])
            mutations.append(mut)
            selection_names.append(name)

cmd.hide("all")
cmd.show("surface")

#for prot in HEADERS_PROT:
#    cmd.alter(prot, "vdw=" + str(0.1))


sr_selname = findSurfaceResidues("all", 2.5, 1)

exposed_res = [tup[1] for tup in sr_selname]
mutation_index = [trip[1] for trip in mutations]
        
exposed_mut = [n for n in mutation_index if n in exposed_res]

print(len(exposed_mut)/len(mutations))

print(mutations)
for name in selection_names:
    cmd.alter(name, "cdw=" + str(2))
    cmd.color("red", name)
