#!/usr/bin/env python

from Bio import motifs
import click
import io
import numpy as np
import os
import pandas as pd
import pickle
import random
random.seed(0)
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
@click.option(
    "-w", "--weights",
    help="Type of weights.",
    type=click.Choice(["PFM", "PWM"], case_sensitive=False),
    default="PFM",
    show_default=True,
)

def main(**args):

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

    # Parse binding modes
    binding_modes = {}
    binding_modes_file = os.path.join(data_dir, "leaf_to_cluster.tab")
    with open(binding_modes_file) as handle:
        for line in handle:
            matrix_id, cluster = line.strip("\n").split("\t")
            matrix_id = matrix_id.split("_")
            binding_modes.setdefault(int(cluster), [])
            matrix_id = "%s.%s" % (matrix_id[-2], matrix_id[-1])
            if matrix_id in profiles:
                binding_modes[int(cluster)].append(matrix_id)

    # For each binding mode...
    filters = {}
    filters_pfms = {}
    for bm in sorted(binding_modes):
        for matrix_id in binding_modes[bm]:
            m = profiles[matrix_id]
            tfs = m.name.upper().split("(")[0].split("::")
            pwm = _PWM_to_filter_weights(m, args["weights"],
                args["filter_size"])
            filters.setdefault("%s;fwd;%s" % (matrix_id, "::".join(tfs)), pwm)
            filters.setdefault("%s;rev;%s" % (matrix_id, "::".join(tfs)),
                pwm[::-1,::-1])
            pfm = _PWM_to_filter_weights(m, filter_size=args["filter_size"])
            filters_pfms.setdefault("%s;fwd;%s" % (matrix_id, "::".join(tfs)),
                pfm)
            filters_pfms.setdefault("%s;rev;%s" % (matrix_id, "::".join(tfs)),
                pfm[::-1,::-1])

    # MEME
    meme_file = os.path.join(base_dir,
        "JASPAR2020_CORE_vertebrates_nr_f=%s.meme" % str(args["filter_size"]))
    if not os.path.exists(meme_file):

        s = ""
        s += "MEME version 4\n\n"
        s += "ALPHABET= ACGT\n\n"
        s += "strands: + -\n\n"
        s += "Background letter frequencies (from uniform background):\n"
        s += "A 0.25000 C 0.25000 G 0.25000 T 0.25000\n\n"

        for name in sorted(filters_pfms):
            m = "\n".join(["\t".join(list(map(str, i+.25))) \
                for i in filters_pfms[name]])
            handle = io.StringIO(m)
            m = motifs.read(handle, "pfm-four-columns")
            consensus_seq = m.consensus
            w = len(consensus_seq)
            s += "MOTIF %s %s\n" % (name, consensus_seq)
            s += "letter-probability matrix: alength= 4 w= %s nsites= 20 E= 0\n" % w
            pwm = [m.pwm[nt] for nt in "ACGT"]
            for row in np.transpose(np.array(pwm)):
                s += " ".join([str(round(r, 8)).rjust(11) for r in row]) + "\n"
            s += "\n"

        with open(meme_file, "w") as handle:
            handle.write(s)

    # TOMTOM
    tomtom_file = os.path.join(base_dir,
        "JASPAR2020_CORE_vertebrates_nr_f=%s.tsv" % str(args["filter_size"]))
    if not os.path.exists(tomtom_file):

        cmd = "tomtom %s %s -thresh 5000 -evalue -norc --text > %s" % (
            meme_file, meme_file, tomtom_file)
        _ = sp.run([cmd], shell=True)

    # Non-redundant filters
    ixs = {}
    tomtom = []
    nr_filters = {}
    df = pd.read_csv(tomtom_file, sep="\t", header=0, usecols=[0, 1, 5],
        comment="#")
    df.sort_values(by=["Query_ID", "Target_ID"], inplace=True)
    matrix_ids = list(df["Query_ID"].unique())
    for ix, matrix_id in enumerate(matrix_ids):
        ixs.setdefault(matrix_id, ix)
        tomtom.append(df[df["Query_ID"] == matrix_id]["q-value"].to_list())
    random.shuffle(matrix_ids)
    while len(nr_filters) < args["n_filters"] and len(matrix_ids) > 0:
        matrix_id = matrix_ids.pop(0)
        is_nr = True
        for nr_matrix_id in nr_filters:
            if tomtom[ixs[nr_matrix_id]][ixs[matrix_id]] <= 0.05:
                is_nr = False
                break
        if is_nr:
            nr_filters.setdefault(matrix_id, filters[matrix_id])

    # Save
    with open(args["out_file"], "wb") as handle:
        pickle.dump(nr_filters, handle, protocol=pickle.HIGHEST_PROTOCOL)

def _PWM_to_filter_weights(motif, weights="PFM", filter_size=19):

    # Initialize
    lpop = 0
    rpop = 0

    # If PFM...
    if weights == "PFM":
        pwm = np.array([list(i) for i in motif.pwm.values()]) - .25
    # ... Else...
    else:
        motif.pseudocounts = motifs.jaspar.calculate_pseudocounts(motif)
        pwm = np.array([list(i) for i in motif.pssm.values()])

    # Extend the PWM with zeros to the left and right
    pwm = np.concatenate((np.zeros((4, filter_size)), pwm,
        np.zeros((4, filter_size))), axis=1)

    # Transpose PWM
    pwm = pwm.T.tolist()

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

    return(np.array(pwm))

if __name__ == "__main__":
    main()