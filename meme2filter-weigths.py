#!/usr/bin/env python

from Bio import motifs
import click
import io
import numpy as np
import os
import pandas as pd
import pickle
import re
from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans
import subprocess as sp

CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
}

@click.command(no_args_is_help=True, context_settings=CONTEXT_SETTINGS)
@click.argument(
    "meme_file",
    type=click.Path(resolve_path=True),
)
@click.argument(
    "pickle_file",
    type=click.Path(resolve_path=True),
)
@click.option(
    "-f", "--filter-size",
    help="Filter size.",
    type=int,
    default=19,
    show_default=True,
)

def main(**params):

    # Initialize
    base_dir = os.path.dirname(os.path.realpath(__file__))

    # Parse profiles
    profiles = {}
    with open(params["meme_file"]) as handle:
        for line in handle:
            line = line.strip("\n")
            if line.startswith("*"):
                skip = True
            elif line.startswith("A "):
                background = line.split()
            elif line.startswith("MOTIF"):
                matrix_id = line[6:]
                profiles.setdefault(matrix_id, [])
                skip = False
            elif line.startswith("letter-probability"):
                # profiles[-1][1] = line
                continue
            else:
                values = re.findall("(\d\.\d+)", line)
                if len(values) == 4 and not skip:
                    profiles[matrix_id].append([])
                    for value in values:
                        profiles[matrix_id][-1].append(float(value))

    # For each profile...
    filters = {}
    for matrix_id in profiles:
        pwm = profiles[matrix_id]
        pwm = _PWM_to_filter_weights(pwm, params["filter_size"])
        filters.setdefault("%s;fwd" % matrix_id, pwm)
        filters.setdefault("%s;rev" % matrix_id, np.flip(pwm))

    # meme_file = os.path.join(base_dir,
    #     "JASPAR2020_CORE_vertebrates_non-redundant_pfms_filter-size=%s.meme" % str(params["filter_size"]))
    # if not os.path.exists(meme_file):

    #     lines = ""
    #     lines += "MEME version 4\n\n"
    #     lines += "ALPHABET= ACGT\n\n"
    #     lines += "strands: + -\n\n"
    #     lines += "Background letter frequencies (from uniform background):\n"
    #     lines += "A 0.25000 C 0.25000 G 0.25000 T 0.25000\n\n"

    #     for name in sorted(filters):
    #         m = "\n".join(["\t".join(list(map(str, i+.25))) for i in filters[name]])
    #         handle = io.StringIO(m)
    #         m = motifs.read(handle, "pfm-four-columns")
    #         consensus_seq = m.consensus
    #         w = len(consensus_seq)
    #         lines += "MOTIF %s %s\n" % (name, consensus_seq)
    #         lines += "letter-probability matrix: alength= 4 w= %s nsites= 20 E= 0\n" % w
    #         pwm = [m.pwm[nt] for nt in "ACGT"]
    #         for row in np.transpose(np.array(pwm)):
    #             lines += " ".join([str(round(r, 8)).rjust(11) for r in row]) + "\n"
    #         lines += "\n"

    #     with open(meme_file, "w") as handle:
    #         handle.write(lines)

    # tomtom_file = os.path.join(base_dir,
    #     "JASPAR2020_CORE_vertebrates_non-redundant_pfms_filter-size=%s.tsv" % str(params["filter_size"]))
    # if not os.path.exists(tomtom_file):

    #     cmd = "tomtom %s %s -thresh 5000 -evalue -norc --text > %s" % (meme_file, meme_file,
    #         tomtom_file)
    #     _ = sp.run([cmd], shell=True)

    # exit(0)

    # # K-means
    # Xs = []
    # ys = []
    # ixs = {}
    # clusters = {}
    # distances = {}
    # df = pd.read_csv(tomtom_file, sep="\t", header=0, usecols=[0, 1, 5], comment="#")
    # df["q-value"] = df["q-value"].apply(lambda x: 0 if x > 0.05 else 1)
    # df.sort_values(by=["Query_ID", "Target_ID"], inplace=True)
    # for i, matrix_id in enumerate(df["Query_ID"].unique()):
    #     ixs.setdefault(matrix_id, i)
    # for y in ixs:
    #     Xs.append([0] * len(ixs))
    #     ys.append(y)
    # for i, row in df.iterrows():
    #     Xs[ixs[row["Query_ID"]]][ixs[row["Target_ID"]]] = row["q-value"]
    # Xs = np.array(Xs)
    # km = KMeans(n_clusters=params["n_filters"], random_state=0).fit(Xs)
    # zipped_list = list(zip(km.labels_, Xs, ys))
    # for z in range(len(zipped_list)):
    #     cluster, X, y = zipped_list[z]
    #     clusters.setdefault(cluster, [])
    #     clusters[cluster].append(y)
    #     distance = cosine(km.cluster_centers_[cluster], X)
    #     distances.setdefault(y, distance)

    # # Subsampled filters
    # sub_filters = {}
    # for c in sorted(clusters):
    #     clusters[c].sort(key=lambda x: distances[x])
    #     matrix_id = clusters[c][0]
    #     sub_filters.setdefault(matrix_id, filters[matrix_id])

    # Save
    with open(params["pickle_file"], "wb") as handle:
        # pickle.dump(sub_filters, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(filters, handle, protocol=pickle.HIGHEST_PROTOCOL)


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