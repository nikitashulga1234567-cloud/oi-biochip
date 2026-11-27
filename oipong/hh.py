"""Нейронная биофизика: Ходжкин-Хаксли (таблицы скоростей) + синапсы-проводимости.
Температура и нейромедиаторы приходят из BioEnvironment (kin_factor, cond_factor, glu/gaba gain)."""
import numpy as np

_VMIN, _VMAX, _RES = -120.0, 80.0, 20.0
_VG = np.arange(_VMIN, _VMAX + 1.0 / _RES, 1.0 / _RES)


def _vtrap(x, y):
    x = np.where(np.abs(x) < 1e-7, 1e-7, x)
    return x / (-np.expm1(-x / y))


_a_m = 0.1 * _vtrap(_VG + 40.0, 10.0)
_b_m = 4.0 * np.exp(-(_VG + 65.0) / 18.0)
_a_h = 0.07 * np.exp(-(_VG + 65.0) / 20.0)
_b_h = 1.0 / (1.0 + np.exp(-(_VG + 35.0) / 10.0))
_a_n = 0.01 * _vtrap(_VG + 55.0, 10.0)
_b_n = 0.125 * np.exp(-(_VG + 65.0) / 80.0)
_S_m, _S_h, _S_n = _a_m + _b_m, _a_h + _b_h, _a_n + _b_n

G_NA, E_NA = 120.0, 50.0
G_K, E_K = 36.0, -77.0
G_L, E_L = 0.3, -54.387
E_EXC, E_INH = 0.0, -80.0
REST = dict(v=-65.0, m=0.053, h=0.596, n=0.318)


def integrate_frame(net, cfg, I_ext, rng, env, explore=False):
    """Один кадр (frame_ms): интегрирует все нейроны. Возвращает число спайков каждого нейрона."""
    from .config import ROLE_SENS, ROLE_HID, ROLE_MOT
    S = int(round(cfg.frame_ms / cfg.dt))
    N = net.N
    mn = cfg.explore_noise if (explore and cfg.explore_noise is not None) else cfg.mot_noise
    mb = cfg.explore_bias if (explore and cfg.explore_bias is not None) else cfg.mot_bias
    hn = cfg.explore_hid_noise if (explore and cfg.explore_hid_noise is not None) else cfg.noise_sigma
    noise = rng.normal(0.0, 1.0, (S // 20 + 1, N))
    scale = np.full(N, cfg.noise_sigma)
    scale[net.role == ROLE_HID] = hn
    scale[net.role == ROLE_MOT] = mn if mn is not None else cfg.noise_sigma
    scale[net.role == ROLE_SENS] = 0.0
    noise *= scale
    bias = np.where(net.role == ROLE_HID, cfg.hid_bias, 0.0) + np.where(net.role == ROLE_MOT, 0.0 if mb is None else mb, 0.0)

    phi = env.kin_factor                    # температурный фактор кинетики m, h, n
    gk = env.cond_factor                    # температурный фактор проводимостей
    v, m, h, n = net.v, net.m, net.h, net.n
    ge, gi, above = net.ge, net.gi, net.above
    counts = np.zeros(N)
    dt = cfg.dt
    sr = env.syn_rate_factor
    de, di = np.exp(-dt * sr / cfg.syn_tau_e), np.exp(-dt * sr / cfg.syn_tau_i)
    gs = cfg.g_scale
    We = gs * env.glu_gain * net.W * (~net.inhib)[:, None]       # возбуждение: глутамат
    Wi = gs * env.gaba_gain * net.W * net.inhib[:, None]         # торможение: ГАМК
    gna, gkk, gl = G_NA * gk, G_K * gk, G_L * gk
    top = len(_VG) - 1
    for s in range(S):
        idx = np.clip(((v - _VMIN) * _RES).astype(int), 0, top)
        m += dt * phi * (_a_m[idx] - _S_m[idx] * m)
        h += dt * phi * (_a_h[idx] - _S_h[idx] * h)
        n += dt * phi * (_a_n[idx] - _S_n[idx] * n)
        i_ion = gna * m ** 3 * h * (v - E_NA) + gkk * n ** 4 * (v - E_K) + gl * (v - E_L)
        i_syn = ge * (E_EXC - v) + gi * (E_INH - v)
        v += dt * (I_ext + bias + noise[s // 20] + i_syn - i_ion)
        np.clip(v, -100.0, 70.0, out=v)
        ge *= de
        gi *= di
        now = v > 0.0
        sp = now & ~above
        above[:] = now
        if sp.any():
            counts += sp
            spf = sp.astype(float)
            ge += spf @ We
            gi += spf @ Wi
    return counts
