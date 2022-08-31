import pandas as pd
import numpy as np


configfile: "config/smk-config.yml"


df = pd.read_csv("config/preproc-filtered.csv")

preproc_to_pfc = expand(config["preproc_to_pfc"], zip, sub=df["sub"], task=df["task"])


rule all:
    input:
        config["agg_pfc"],


rule agg_pfc:
    input:
        preproc_to_pfc,
    output:
        config["agg_pfc"],
    run:
        runs = np.stack([np.load(vector) for vector in input], axis=0)
        assert len(runs) == 80, "missing runs"
        subjects = (runs[::2] + runs[1::2]) / 2
        np.save(output[0], subjects)


rule preproc_to_pfc:
    input:
        cmd="scripts/00-preproc-to-pfc.py",
        path=config["preproc"],
    output:
        config["preproc_to_pfc"],
    params:
        smooth_fwhm=config["smooth_fwhm"],
        denoise_strategy=config["denoise_strategy"],
    threads: 1
    shell:
        "python {input.cmd} {input.path} {output} {params.denoise_strategy} --smooth_fwhm {params.smooth_fwhm} --n_jobs {threads}"
