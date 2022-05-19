#!/usr/bin/env python

from Bio import motifs
import click
import io
import numpy as np
import os
import pandas as pd
import pickle
import subprocess as sp
from tqdm import tqdm
bar_format = "{percentage:3.0f}%|{bar:20}{r_bar}"

CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
}

@click.command(no_args_is_help=True, context_settings=CONTEXT_SETTINGS)
@click.argument(
    "out_file",
    type=click.Path(resolve_path=True),
)
@click.option(
    "-e", "--exclude-tf",
    help="Do not include motifs from given TF.",
    multiple=True,
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
    help="Number of filters to subsample.  [default: max]",
    type=int,
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
    excluded_motifs = set()
    exclude_tfs = set(args["exclude_tf"])
    base_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(base_dir, "data")

    # Parse profiles
    profiles = {}
    jaspar_file = os.path.join(data_dir,
        "JASPAR2020_CORE_vertebrates_non-redundant_pfms_jaspar.txt")
    with open(jaspar_file) as handle:
        for m in motifs.parse(handle, "jaspar"):
            profiles.setdefault(m.matrix_id, m)

    # For each binding mode...
    filters = {}
    filters_pfms = {}
    for matrix_id in profiles:
        m = profiles[matrix_id]
        tfs = m.name.upper().split("(")[0].split("::")
        # Exclude TF
        if exclude_tfs.intersection(set(tfs)):
            excluded_motifs.add(matrix_id)
            continue
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
    nr_filters = {}
    tomtom = pd.read_csv(tomtom_file, sep="\t", header=0, usecols=[0, 1, 5],
        comment="#")
    tomtom.sort_values(by=["Query_ID", "Target_ID"], inplace=True)
    if args["n_filters"] is None:
        n_filters = len(filters)
    else:
        n_filters = args["n_filters"]
    for i in tqdm(range(n_filters), total=n_filters, bar_format=bar_format):
        nr_filter = _get_nr_filter(tomtom, set(list(nr_filters.keys())),
            excluded_motifs)
        nr_filters.setdefault(nr_filter, filters[nr_filter])
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

def _get_nr_filter(tomtom, nr_filters, excluded_motifs):

    # First filter...
    if len(nr_filters) == 0:
        df = tomtom.groupby(["Query_ID"]).sum().\
                    sort_values(by=["q-value"], ascending=False)
        for matrix_id in df.index:
            # Exclude TF
            l = matrix_id.split(";")
            if l[0] in excluded_motifs:
                continue
            return(matrix_id)
    # ... Other filters...
    else:
        df = tomtom[tomtom["Target_ID"].isin(list(nr_filters))].\
                                        groupby(["Query_ID"]).sum().\
                                        sort_values(by=["q-value"], ascending=False)
        for matrix_id in df.index:
            # Exclude TF
            l = matrix_id.split(";")
            if l[0] in excluded_motifs:
                continue
            if matrix_id not in nr_filters:
                return(matrix_id)

if __name__ == "__main__":
    main()