#!/usr/bin/env python

from xml.dom import EMPTY_PREFIX
from Bio import motifs
import click
import io
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.cluster import KMeans
import subprocess as sp

CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
}

@click.command(no_args_is_help=True, context_settings=CONTEXT_SETTINGS)
@click.argument(
    "binding_modes_pickle",
    type=click.Path(resolve_path=True),
)
@click.argument(
    "out_file",
    type=click.Path(resolve_path=True),
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
    prefix = os.path.splitext(os.path.basename(params["binding_modes_pickle"]))[0]

    # Load
    probs = {}
    matrix_id2filter = {}
    ixs = {"A":0, "C":1, "G":2, "T":3}
    with open(params["binding_modes_pickle"], "rb") as handle:
        filters = pickle.load(handle)

    meme_file = os.path.join(base_dir, "%s.meme" % prefix)

    lines = ""
    lines += "MEME version 4\n\n"
    lines += "ALPHABET= ACGT\n\n"
    lines += "strands: + -\n\n"
    lines += "Background letter frequencies (from uniform background):\n"
    lines += "A 0.25000 C 0.25000 G 0.25000 T 0.25000\n\n"

    for bm in sorted(filters):
        m = "\n".join(["\t".join(list(map(str, i+.25))) for i in filters[bm][0][2]])
        handle = io.StringIO(m)
        m = motifs.read(handle, "pfm-four-columns")
        name = "%s;fwd;%s" % (filters[bm][0][0], "::".join(filters[bm][0][1]))
        consensus_seq = m.consensus
        w = len(consensus_seq)
        lines += "MOTIF %s %s\n" % (name, consensus_seq)
        lines += "letter-probability matrix: alength= 4 w= %s nsites= 20 E= 0\n" % w
        for row in np.transpose(np.array(list(m.pwm.values()))):
            lines += " ".join([str(round(r, 8)).rjust(11) for r in row]) + "\n"
        lines += "\n"
        matrix_id2filter.setdefault(name, filters[bm][0][2])
        prob = sum([m.pwm[ixs[consensus_seq[i]]][i] for i in range(len(consensus_seq))])
        probs.setdefault(name, prob)
        m = "\n".join(["\t".join(list(map(str, i+.25))) for i in filters[bm][0][3]])
        handle = io.StringIO(m)
        m = motifs.read(handle, "pfm-four-columns")
        name = "%s;rev;%s" % (filters[bm][0][0], "::".join(filters[bm][0][1]))
        consensus_seq = m.consensus
        w = len(consensus_seq)
        lines += "MOTIF %s %s\n" % (name, consensus_seq)
        lines += "letter-probability matrix: alength= 4 w= %s nsites= 20 E= 0\n" % w
        for row in np.transpose(np.array(list(m.pwm.values()))):
            lines += " ".join([str(round(r, 8)).rjust(11) for r in row]) + "\n"
        lines += "\n"
        matrix_id2filter.setdefault(name, filters[bm][0][3])
        prob = sum([m.pwm[ixs[consensus_seq[i]]][i] for i in range(len(consensus_seq))])
        probs.setdefault(name, prob)

    with open(meme_file, "w") as handle:
        handle.write(lines)

    tomtom_dir = os.path.join(base_dir, prefix)
    if not os.path.isdir(tomtom_dir):

        cmd = "tomtom %s %s -thresh 1000 -evalue -norc -o %s" % (meme_file,
            meme_file, tomtom_dir)
        _ = sp.run([cmd], shell=True)

    df = pd.read_csv(os.path.join(tomtom_dir, "tomtom.tsv"), sep="\t", header=0,
        usecols=[0, 1, 3], comment="#")
    df["p-value"] = df["p-value"].apply(lambda x: 0 if x > 0.05 else 1)
    df.sort_values(by=["Query_ID", "Target_ID"], inplace=True)

    Xs = []
    ys = []
    for matrix_id in df["Query_ID"].unique():
        log10s = df[df["Query_ID"] == matrix_id]["p-value"].to_list()
        Xs.append(log10s)
        ys.append(matrix_id)
    Xs = np.array(Xs)

    # Get clusters
    clusters = {}
    kmeans = KMeans(n_clusters=params["n_filters"], random_state=0).fit(Xs)
    for z in zip(kmeans.labels_, ys):
        clusters.setdefault(z[0], [])
        clusters[z[0]].append(z[1])

    # Subsampled filters
    sub_filters = {}
    for c in sorted(clusters):
        clusters[c].sort(key=lambda x: probs[x], reverse=True)
        matrix_id = clusters[c][0]
        sub_filters.setdefault(matrix_id, matrix_id2filter[matrix_id])

    # Save
    with open(params["out_file"], "wb") as handle:
        pickle.dump(sub_filters, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()