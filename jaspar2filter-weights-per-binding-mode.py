#!/usr/bin/env python

from Bio import motifs
import click
import numpy as np
import os
import pickle

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
    for bm in sorted(binding_modes):
        for matrix_id in binding_modes[bm]:
            m = profiles[matrix_id]
            filters.setdefault(bm, [])
            pwm = _PWM_to_filter_weights(m, args["weights"],
                args["filter_size"])
            filters[bm].append([matrix_id, pwm, pwm[::-1,::-1]])

    # Save
    with open(args["out_file"], "wb") as handle:
        pickle.dump(filters, handle, protocol=pickle.HIGHEST_PROTOCOL)

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