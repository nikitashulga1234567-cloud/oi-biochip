"""oipong v4: 3D-симулятор органоидного интеллекта (биочип на HD-MEA).

Движок v3 (Pong, без изменений интерфейсов):
  config, hh (Ходжкин–Хаксли 1952), environment (BioEnvironment: Q10, глутамат, ГАМК, дофамин), network, game,
  learning (RM-Hebb), sim (OIPong), experiments.
Движок уровня 2 (новое в v4):
  bioconfig (BioConfig), neuron (Pospischil et al. 2008: Na m^3h, Kd n^4, M-ток p, утечка; RS/FS),
  gel (температура Q10, блок NMDA Mg2+ по Jahr & Stevens 1990, вязкость -> радиус связности, диффузия K+),
  mea (HD-MEA: считывание с затуханием, гауссова стимуляция), bionet (3D-сеть, AMPA/NMDA/ГАМК-А),
  plasticity (STDP + след пригодности + дофамин + синаптическое масштабирование), task_ab (дискриминация A/B),
  info (MI, TE), viz (панель V1–V6, анимация; импортируется отдельно: from oipong import viz).
"""
__version__ = "4.0"

from .config import (Config, PRESETS, make_config, cfg_to_dict, cfg_from_dict, save_config, load_config, log_run,
                     ROLE_SENS, ROLE_HID, ROLE_MOT, ROLE_NAME)
from .environment import BioEnvironment
from .sim import OIPong, save_game, load_game
from .experiments import (baseline_winrate, make_oracle, weight_structure, benchmark, ablation, debug_ladder,
                          temperature_scan, dopamine_scan)
# уровень 2
from .bioconfig import BioConfig, make_bio_config, bio_cfg_to_dict, save_bio_config
from .neuron import NeuronArray, RS, FS, TYPE_NAME, CELL_PARAMS, fi_curve, simulate_single
from .gel import GelMedium, mg_block, q10_factor, conn_radius_from_viscosity
from .mea import HDMEA
from .bionet import BioNet, Recorder
from .plasticity import RSTDP
from .task_ab import ABTask, run_ab_experiment, dprime
from .info import mutual_information, transfer_entropy, mi_with_surrogate, te_with_surrogate, sensory_motor_info
import numpy as np
BOX = np.array(Config.box)   # для совместимости с ячейками визуализации v3 (размер матрицы в cfg.box)
