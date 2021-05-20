#!/usr/bin/env python

import enum
from Bio import motifs
import click
import io
import numpy as np
import os
import pandas as pd
import pickle
import random
random.seed(0)
from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans
import subprocess as sp

CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
}

@click.command(no_args_is_help=True, context_settings=CONTEXT_SETTINGS)
@click.argument(
    "out_file",
    type=click.Path(resolve_path=True),
)
@click.option(
    "-f", "--filter-size",
    help="Filter size.",
    type=int,
    default=19,
    show_default=True,
)
@click.option(
    "-n", "--n-filters",
    help="Number of filters to subsample.",
    type=int,
    default=100,
    show_default=True,
)

def main(**params):

    # Initialize
    base_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(base_dir, "data")

    # Parse profiles
    profiles = {}
    jaspar_file = os.path.join(data_dir,
        "JASPAR2020_CORE_vertebrates_non-redundant_pfms_jaspar.txt")
    with open(jaspar_file) as handle:
        for m in motifs.parse(handle, "jaspar"):
            profiles.setdefault(m.matrix_id, m)

    # For each profile...
    filters = {}
    for matrix_id in profiles:
        m = profiles[matrix_id]
        tfs = m.name.upper().split("(")[0].split("::")
        pwm = [list(i) for i in m.pwm.values()]
        pwm = _PWM_to_filter_weights(list(map(list, zip(*pwm))), params["filter_size"])
        filters.setdefault("%s;fwd;%s" % (matrix_id, "::".join(tfs)), pwm)
        filters.setdefault("%s;rev;%s" % (matrix_id, "::".join(tfs)), np.flip(pwm))

    # MEME
    meme_file = os.path.join(base_dir,
        "JASPAR2020_CORE_vertebrates_non-redundant_pfms_filter-size=%s.meme" % str(params["filter_size"]))
    if not os.path.exists(meme_file):

        lines = ""
        lines += "MEME version 4\n\n"
        lines += "ALPHABET= ACGT\n\n"
        lines += "strands: + -\n\n"
        lines += "Background letter frequencies (from uniform background):\n"
        lines += "A 0.25000 C 0.25000 G 0.25000 T 0.25000\n\n"

        for name in sorted(filters):
            m = "\n".join(["\t".join(list(map(str, i+.25))) for i in filters[name]])
            handle = io.StringIO(m)
            m = motifs.read(handle, "pfm-four-columns")
            consensus_seq = m.consensus
            w = len(consensus_seq)
            lines += "MOTIF %s %s\n" % (name, consensus_seq)
            lines += "letter-probability matrix: alength= 4 w= %s nsites= 20 E= 0\n" % w
            pwm = [m.pwm[nt] for nt in "ACGT"]
            for row in np.transpose(np.array(pwm)):
                lines += " ".join([str(round(r, 8)).rjust(11) for r in row]) + "\n"
            lines += "\n"

        with open(meme_file, "w") as handle:
            handle.write(lines)

    # TOMTOM
    tomtom_file = os.path.join(base_dir,
        "JASPAR2020_CORE_vertebrates_non-redundant_pfms_filter-size=%s.tsv" % str(params["filter_size"]))
    if not os.path.exists(tomtom_file):

        cmd = "tomtom %s %s -thresh 5000 -evalue -norc --text > %s" % (meme_file, meme_file,
            tomtom_file)
        _ = sp.run([cmd], shell=True)

    # Non-redundant filters
    ixs = {}
    tomtom = []
    nr_filters = {}
    df = pd.read_csv(tomtom_file, sep="\t", header=0, usecols=[0, 1, 5], comment="#")
    df.sort_values(by=["Query_ID", "Target_ID"], inplace=True)
    matrix_ids = list(df["Query_ID"].unique())
    for ix, matrix_id in enumerate(matrix_ids):
        ixs.setdefault(matrix_id, ix)
        tomtom.append(df[df["Query_ID"] == matrix_id]["q-value"].to_list())
    random.shuffle(matrix_ids)
    while len(nr_filters) < params["n_filters"] and len(matrix_ids) > 0:
        matrix_id = matrix_ids.pop(0)
        is_nr = True
        for nr_matrix_id in nr_filters:
            if tomtom[ixs[nr_matrix_id]][ixs[matrix_id]] <= 0.05:
                is_nr = False
                break
        if is_nr:
            nr_filters.setdefault(matrix_id, filters[matrix_id])

    # Save
    with open(params["out_file"], "wb") as handle:
        pickle.dump(nr_filters, handle, protocol=pickle.HIGHEST_PROTOCOL)

def _PWM_to_filter_weights(pwm, filter_size=19):

    # Initialize
    lpop = 0
    rpop = 0

    pwm = [[.25,.25,.25,.25]]*filter_size+pwm+[[.25,.25,.25,.25]]*filter_size

    while len(pwm) > filter_size:
        if max(pwm[0]) < max(pwm[-1]):
            pwm.pop(0)
            lpop += 1
        elif max(pwm[-1]) < max(pwm[0]):
            pwm.pop(-1)
            rpop += 1
        else:
            if lpop > rpop:
                pwm.pop(-1)
                rpop += 1
            else:
                pwm.pop(0)
                lpop += 1

    return(np.array(pwm) - .25)

if __name__ == "__main__":
    main()