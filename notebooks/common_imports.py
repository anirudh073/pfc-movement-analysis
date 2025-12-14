import os
import datajoint as dj
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if os.path.basename(os.getcwd()) == 'notebooks':
    os.chdir("..")
    
dj.config.load("dj_local_conf.json")

base_dir = "/media/labuser/NA_1_2025/spyglass/wilbur/"
raw_dir = base_dir + "raw/"

os.environ["SPYGLASS_BASE_DIR"] = base_dir
os.environ["SPYGLASS_RAW_DIR"] = raw_dir

import spyglass.common as sgc

from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename
nwb_file_name = "wilbur20210512.nwb"
nwb_copy_file_name = get_nwb_copy_filename(nwb_file_name)
session_restrict = {"nwb_file_name": nwb_copy_file_name}