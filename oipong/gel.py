"""Гелевая среда биочипа: температура (Q10), магний и блок NMDA, вязкость -> радиус связности,
диффузия внеклеточного K+ на сетке вокселей.

1. Температура (Q10). Скорость процесса при T: k(T) = k(T_ref) * Q10 ** ((T - T_ref) / 10).
   - затворы m, h, n, p: Q10_gate = 3 (Hodgkin & Huxley 1952; Hille 2001: 2–4 для воротных процессов);
   - максимальные проводимости: Q10_cond ~ 1.3 (Hille 2001: проводимость одиночного канала 1.2–1.5);
     обоснование: проводимость ограничена диффузией ионов в поре, её Q10 близок к Q10 диффузии (~1.3);
   - спад синаптических проводимостей: Q10_syn = 2 (допущение).
   T_ref = 36 °C: при ней даны кинетики Pospischil et al. 2008.

2. Блок NMDA ионом Mg2+ (Jahr & Stevens 1990, J. Neurosci. 10:3178):
       B(V) = 1 / (1 + [Mg2+]o / 3.57 мМ * exp(-0.062 V))         V в мВ
   При [Mg] = 1 мМ и V = -70 мВ B ~ 0.04 (канал почти закрыт), при V = 0 мВ B ~ 0.78.

3. Вязкость геля -> предел радиуса связности (ДОПУЩЕНИЕ, феноменология):
       conn_radius = conn_radius_ref * (eta_ref / eta) ** visc_exponent
   Качественная опора: в более плотной/жёсткой 3D-матрице нейриты растут медленнее и короче
   (Balgude et al. 2001, Biomaterials 22:1077, агарозные гели). Количественной формулы в литературе мы не нашли.

4. Внеклеточный K+: каждый спайк выбрасывает k_per_spike_fmol K+ в воксель нейрона;
       d[K]/dt = D* лапласиан([K]) - ([K] - K_rest) / tau_clear + выброс / (alpha * V_voxel)
   D* = D / lambda^2 (извилистость, Nicholson & Syková 1998). Граничное условие: нулевой поток.
   Сдвиг потенциала реверсии калия по Нернсту:  E_K = E_K0 + (RT/F) ln([K]o / K_rest),
   RT/F = 26.7 мВ при 36 °C. Утечка не сдвигается (упрощение).
"""
import numpy as np

R_GAS, F_FARADAY = 8.314, 96485.0


def q10_factor(q10, T, T_ref):
    """Q10 ** ((T - T_ref) / 10)."""
    return float(q10 ** ((T - T_ref) / 10.0))


def mg_block(V, mg_mM=1.0):
    """Доля незаблокированных NMDA-каналов, Jahr & Stevens 1990."""
    return 1.0 / (1.0 + (mg_mM / 3.57) * np.exp(-0.062 * V))


def conn_radius_from_viscosity(cfg):
    """Радиус связности (мкм) при относительной вязкости cfg.viscosity. Допущение, см. модуль."""
    return float(cfg.conn_radius_ref * (1.0 / max(cfg.viscosity, 1e-6)) ** cfg.visc_exponent)


class GelMedium:
    """Состояние среды: температура, магний, поле [K+]o."""

    def __init__(self, cfg, xyz=None):
        self.cfg = cfg
        self.T = float(cfg.temperature)
        self.mg = float(cfg.mg_mM)
        self.conn_radius = conn_radius_from_viscosity(cfg)
        self._init_k(xyz)

    # -------- температура
    @property
    def phi_gate(self):
        return q10_factor(self.cfg.q10_gate, self.T, self.cfg.temp_ref)

    @property
    def phi_cond(self):
        return q10_factor(self.cfg.q10_cond, self.T, self.cfg.temp_ref)

    @property
    def phi_syn(self):
        return q10_factor(self.cfg.q10_syn, self.T, self.cfg.temp_ref)

    @property
    def rt_f(self):
        """RT/F в мВ при текущей температуре."""
        return 1000.0 * R_GAS * (self.T + 273.15) / F_FARADAY

    def nmda_block(self, V):
        return mg_block(V, self.mg)

    # -------- калий
    def _init_k(self, xyz):
        c = self.cfg
        self.k_shape = tuple(int(np.ceil(b / c.k_voxel)) for b in c.box)
        self.K = np.full(self.k_shape, c.k_rest)
        self.vox = None
        if xyz is not None:
            self.vox = np.clip((xyz / c.k_voxel).astype(int), 0, np.array(self.k_shape) - 1)
            self.vox_flat = np.ravel_multi_index(self.vox.T, self.k_shape)
        # концентрация от одного спайка в одном вокселе, мМ: фмоль / (alpha * V_voxel)
        vol_L = (c.k_voxel * 1e-6) ** 3 * 1e3            # мкм -> м, м3 -> л
        self.k_per_spike_mM = c.k_per_spike_fmol * 1e-15 / (c.k_alpha * vol_L) * 1e3
        self.D_eff = c.k_D / c.k_tortuosity ** 2         # мкм2/мс

    def k_step(self, spike_counts, dt_ms):
        """Шаг диффузии K+ за dt_ms (явная схема; устойчива при D*dt/h^2 < 1/6, у нас ~1e-4)."""
        c = self.cfg
        if not c.k_enabled or self.vox is None:
            return
        if spike_counts is not None and spike_counts.any():
            add = np.bincount(self.vox_flat, weights=spike_counts, minlength=self.K.size)
            self.K += add.reshape(self.k_shape) * self.k_per_spike_mM
        K = self.K
        P = np.pad(K, 1, mode="edge")                     # нулевой поток на границах
        lap = (P[2:, 1:-1, 1:-1] + P[:-2, 1:-1, 1:-1] + P[1:-1, 2:, 1:-1] + P[1:-1, :-2, 1:-1] +
               P[1:-1, 1:-1, 2:] + P[1:-1, 1:-1, :-2] - 6.0 * K) / c.k_voxel ** 2
        self.K = K + dt_ms * (self.D_eff * lap - (K - c.k_rest) / c.k_tau_clear)

    def ek_shift(self):
        """Сдвиг E_K (мВ) для каждого нейрона: (RT/F) ln([K]o / K_rest)."""
        if self.vox is None or not self.cfg.k_enabled:
            return 0.0
        k_here = self.K.ravel()[self.vox_flat]
        return self.rt_f * np.log(np.maximum(k_here, 1e-3) / self.cfg.k_rest)

    def snapshot(self):
        return dict(T=self.T, phi_gate=round(self.phi_gate, 3), phi_cond=round(self.phi_cond, 3), Mg_mM=self.mg,
                    conn_radius=round(self.conn_radius, 1), K_mean=round(float(self.K.mean()), 4),
                    K_max=round(float(self.K.max()), 4))
