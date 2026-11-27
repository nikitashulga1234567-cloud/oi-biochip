"""Правило обучения RM-Hebb (дофамин-модулируемая пластичность).

    E  <- e_decay * E + outer(pre, post - <post>)          # след приоритета
    W  <- clip(W + lr_eff * da * E)                        # da = награда (- baseline, если use_baseline)

Почему это работает там, где v2.1 не работало:
  1. da = награда за эффект ДВИЖЕНИЯ ракетки: при «не двигался» da = 0 -> нулевой baseline, и веса
     НЕ дрейфуют (в v2.1 скользящее среднее награды делало da != 0 в каждом кадре -> дрейф);
  2. награда приходит в том же кадре, что и действие, поэтому след берёт спайки ТЕКУЩЕГО кадра и не
     хранит прошлое (e_decay = 0): иначе вина за действие размазывается на соседние кадры;
  3. пост-активность центрируется (post - <post>): случайные колебания дофамина вычитаются,
     остаётся ковариация «шум исследования <-> награда» (оценка градиента, REINFORCE).
"""
import numpy as np
from .config import ROLE_HID


def apply_plasticity(net, cfg, spikes, prev_spk, da, lr_gain=1.0):
    dev = spikes - net.pavg
    net.pavg = (1.0 - cfg.pavg_tau) * net.pavg + cfg.pavg_tau * spikes
    pre = cfg.pre_cur_w * spikes + cfg.pre_prev_w * prev_spk
    net.E = cfg.e_decay * net.E + np.outer(pre, dev) * net.mask
    if abs(da) < cfg.da_eps or lr_gain <= 0.0:
        return
    lr = np.full(net.N, cfg.lr * lr_gain)
    lr[net.role == ROLE_HID] *= cfg.lr_hid_mult          # скорость для синапсов НА скрытые нейроны
    net.W += lr[None, :] * da * net.E
    net.W = np.clip(net.W, 0.0, cfg.w_max) * net.mask
    net.scale_inputs()
