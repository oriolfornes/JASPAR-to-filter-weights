#!/usr/bin/env python

import click
import gzip
import numpy as np
import pickle
import re

CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
}

@click.command(no_args_is_help=True, context_settings=CONTEXT_SETTINGS)
@click.argument(
    "streme_file",
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

    # Parse profiles
    profiles = {}
    if params["streme_file"].endswith(".gz"):
        handle = gzip.open(params["streme_file"], "rt")
    else:
        handle = open(params["streme_file"])
    for line in handle:
        line = line.strip("\n")
        if line.startswith("*"):
            skip = True
        elif line.startswith("A "):
            continue
        elif line.startswith("MOTIF"):
            matrix_id = line[6:]
            profiles.setdefault(matrix_id, [])
            skip = False
        elif line.startswith("letter-probability"):
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

    # Save
    with open(params["pickle_file"], "wb") as handle:
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