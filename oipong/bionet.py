"""BioNet: 3D-сеть нейронов Pospischil (RS/FS) над HD-MEA в гелевой среде.

Сборка: позиции в объёме box, типы RS/FS, связность по расстоянию (радиус режет гель), дальние аксоны,
синапсы-проводимости AMPA/NMDA/ГАМК-А с задержкой 1 мс, пуассоновский фон, поле K+.
Симуляция идёт кусками (run), между кусками задача (task_ab.py) меняет стимул и выдаёт награду.

Синаптические токи (мкА/см2), суммарные проводимости на постсинаптическом нейроне:
    I_syn = g_AMPA (V - 0) + g_NMDA B(V) (V - 0) + g_GABA (V - E_GABA)
    dg/dt = -g / tau   (+ W_ij при приходе спайка от i через syn_delay)
B(V) — блок Mg2+ (Jahr & Stevens 1990, gel.py). Тип синапса задаёт пресинаптический нейрон:
RS -> AMPA + NMDA (nmda_ratio * W), FS -> ГАМК-А (закон Дейла).
"""
import numpy as np
from .neuron import NeuronArray, RS, FS
from .gel import GelMedium
from .mea import HDMEA


class Recorder:
    """Что записывать во время run(). Всё опционально, чтобы обучение шло быстро."""

    def __init__(self, spikes=True, bin_ms=5.0, v_every_ms=None, gate_ids=(), gate_every=4, mea=True, k_every_ms=None,
                 balance=True):
        self.want_spikes, self.bin_ms, self.v_every_ms = spikes, bin_ms, v_every_ms
        self.gate_ids, self.gate_every, self.want_mea = list(gate_ids), gate_every, mea
        self.k_every_ms, self.want_balance = k_every_ms, balance
        self.reset()

    def reset(self):
        self.spk_t, self.spk_i = [], []
        self.bins_t, self.bin_counts, self.mea_bins = [], [], []
        self.v_t, self.v_snap = [], []
        self.gate_t, self.gates = [], {k: [] for k in ("V", "m", "h", "n", "p")}
        self.k_t, self.k_snap = [], []
        self.bal_t, self.bal_exc, self.bal_inh = [], [], []
        self.stim_label = []          # (t0, t1, label) — отметки стимулов для графиков

    def arrays(self):
        out = dict(spk_t=np.array(self.spk_t), spk_i=np.array(self.spk_i, dtype=int),
                   bins_t=np.array(self.bins_t), bin_counts=np.array(self.bin_counts),
                   mea_bins=np.array(self.mea_bins), v_t=np.array(self.v_t), v_snap=np.array(self.v_snap),
                   gate_t=np.array(self.gate_t), gates={k: np.array(v) for k, v in self.gates.items()},
                   k_t=np.array(self.k_t), k_snap=np.array(self.k_snap), bal_t=np.array(self.bal_t),
                   bal_exc=np.array(self.bal_exc), bal_inh=np.array(self.bal_inh), stim_label=list(self.stim_label),
                   gate_ids=list(self.gate_ids), bin_ms=self.bin_ms)
        return out


class BioNet:
    def __init__(self, cfg, rng=None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng(cfg.seed)
        self._build()
        self.gel = GelMedium(cfg, self.xyz)
        self.mea = HDMEA(cfg, self.xyz)
        self._connect()
        self.nrn = NeuronArray(self.types, dt=cfg.dt, fs_gM=cfg.fs_gM)
        self.apply_environment()
        N = self.N
        self.g_ampa, self.g_nmda, self.g_gaba = np.zeros(N), np.zeros(N), np.zeros(N)
        self.D = int(round(cfg.syn_delay / cfg.dt))
        self.buf = np.zeros((max(self.D, 1), N), dtype=bool)
        self.t = 0.0
        self.step_i = 0
        self.plasticity = None       # объект RSTDP (plasticity.py) или None
        self.I_bias = np.zeros(N)

    # ------------------------------------------------------------ сборка
    def _build(self):
        c, rng = self.cfg, self.rng
        N = c.n_neurons
        self.N = N
        box = np.array(c.box)
        self.box = box
        self.xyz = rng.uniform([5, 5, 5], box - 5, size=(N, 3))
        self.types = np.where(rng.random(N) < c.frac_fs, FS, RS)
        self.is_exc = self.types == RS
        self.is_inh = ~self.is_exc

    def _connect(self):
        """p_ij = p0[класс] exp(-d/lambda) при d <= conn_radius (из вязкости геля) + n_long дальних
        RS->RS аксонов с p ~ exp(-d/long_spread) (допущение, учебник гл. 4)."""
        c, rng, N = self.cfg, self.rng, self.N
        d = np.linalg.norm(self.xyz[:, None, :] - self.xyz[None, :, :], axis=2)
        self.dist = d
        R = self.gel.conn_radius
        E, I = self.is_exc, self.is_inh
        p0 = np.where(np.outer(E, E), c.p0_ee, 0.0) + np.where(np.outer(E, I), c.p0_ei, 0.0) + \
            np.where(np.outer(I, E), c.p0_ie, 0.0) + np.where(np.outer(I, I), c.p0_ii, 0.0)
        p = p0 * np.exp(-d / c.conn_lambda) * (d <= R)
        np.fill_diagonal(p, 0.0)
        mask = rng.random((N, N)) < p
        # дальние аксоны RS -> RS за пределами радиуса
        exc_idx = np.where(E)[0]
        self.long_mask = np.zeros((N, N), dtype=bool)
        if c.n_long > 0:
            for i in exc_idx:
                cand = exc_idx[(d[i, exc_idx] > R)]
                if len(cand) == 0:
                    continue
                w = np.exp(-d[i, cand] / c.long_spread)
                k = min(c.n_long, len(cand))
                tg = rng.choice(cand, size=k, replace=False, p=w / w.sum())
                self.long_mask[i, tg] = True
        mask |= self.long_mask
        self.mask = mask
        W = np.zeros((N, N))
        wcls = np.where(np.outer(E, E), c.w_ee, 0.0) + np.where(np.outer(E, I), c.w_ei, 0.0) + \
            np.where(np.outer(I, E), c.w_ie, 0.0) + np.where(np.outer(I, I), c.w_ii, 0.0)
        W[mask] = (wcls * rng.uniform(0.5, 1.5, (N, N)))[mask]
        self.W = W
        self.plastic = mask & np.outer(E, E)            # пластичны только E->E
        self.W0 = W.copy()

    def apply_environment(self):
        """Пересчитать температурные множители (после смены gel.T)."""
        g, c = self.gel, self.cfg
        self.nrn.set_temperature_factors(g.phi_gate, g.phi_cond)
        dt = c.dt * g.phi_syn
        self.dec_ampa, self.dec_nmda, self.dec_gaba = (np.exp(-dt / c.tau_ampa), np.exp(-dt / c.tau_nmda),
                                                      np.exp(-dt / c.tau_gaba))
        self.p_bg = 1.0 - np.exp(-c.bg_rate * c.dt / 1000.0)
        self.bg_w = np.where(self.is_exc, c.bg_w, c.bg_w * c.bg_w_fs_mult)

    def set_temperature(self, T):
        self.gel.T = float(T)
        self.apply_environment()

    # ------------------------------------------------------------ симуляция
    def run(self, dur_ms, I_ext=None, rec=None, learn=False):
        """Интегрирует сеть dur_ms. I_ext: ток (мкА/см2, вектор N) на весь кусок. Возвращает число спайков
        каждого нейрона за кусок. learn: передавать спайки в правило пластичности (self.plasticity)."""
        c, nrn, rng = self.cfg, self.nrn, self.rng
        N, dt = self.N, c.dt
        steps = int(round(dur_ms / dt))
        I = (np.zeros(N) if I_ext is None else I_ext) + self.I_bias
        counts = np.zeros(N)
        k_every = max(1, int(round(c.k_update_ms / dt)))
        k_acc = np.zeros(N)
        bin_steps = int(round(rec.bin_ms / dt)) if rec else 0
        v_steps = int(round(rec.v_every_ms / dt)) if (rec and rec.v_every_ms) else 0
        kk_steps = int(round(rec.k_every_ms / dt)) if (rec and rec.k_every_ms) else 0
        bin_acc = np.zeros(N)
        bal_e = bal_i = 0.0
        exc_w_rows = self.is_exc
        plast = self.plasticity if learn else None
        Wexc = self.W[exc_w_rows]
        exc_rows = np.where(exc_w_rows)[0]
        pos_in_exc = -np.ones(N, dtype=int)
        pos_in_exc[exc_rows] = np.arange(len(exc_rows))
        inh_rows = np.where(~exc_w_rows)[0]
        pos_in_inh = -np.ones(N, dtype=int)
        pos_in_inh[inh_rows] = np.arange(len(inh_rows))
        Winh = self.W[inh_rows]
        ek_shift = self.gel.ek_shift()
        nrn.EK_shift = ek_shift if np.ndim(ek_shift) else np.zeros(N)
        for s in range(steps):
            # 1) спад синаптических проводимостей
            self.g_ampa *= self.dec_ampa
            self.g_nmda *= self.dec_nmda
            self.g_gaba *= self.dec_gaba
            # 2) пуассоновский фон (внешние AMPA-события)
            bg = rng.random(N) < self.p_bg
            if bg.any():
                self.g_ampa += bg * self.bg_w
            # 3) приход спайков, выпущенных syn_delay назад
            slot = self.step_i % self.D
            arr = self.buf[slot]
            if arr.any():
                idx = np.nonzero(arr)[0]
                ie = pos_in_exc[idx]
                ie = ie[ie >= 0]
                if len(ie):
                    inc = Wexc[ie].sum(axis=0)
                    self.g_ampa += inc
                    self.g_nmda += c.nmda_ratio * inc
                ii = pos_in_inh[idx]
                ii = ii[ii >= 0]
                if len(ii):
                    self.g_gaba += Winh[ii].sum(axis=0)
            # 4) нейроны
            B = self.gel.nmda_block(nrn.V)
            gn = self.g_nmda * B
            sp = nrn.step(I, self.g_ampa, gn, self.g_gaba, c.E_gaba)
            self.buf[slot] = sp
            self.t += dt
            self.step_i += 1
            if sp.any():
                counts += sp
                k_acc += sp
                if plast is not None:
                    plast.on_spikes(sp, self.t)
            if plast is not None:
                plast.decay_traces()
            if (s + 1) % k_every == 0:
                self.gel.k_step(k_acc, c.k_update_ms)
                k_acc[:] = 0.0
                ek_shift = self.gel.ek_shift()
                if np.ndim(ek_shift):
                    nrn.EK_shift = ek_shift
            if rec is not None:
                if rec.want_spikes and sp.any():
                    ids = np.nonzero(sp)[0]
                    rec.spk_t.extend([self.t] * len(ids))
                    rec.spk_i.extend(ids.tolist())
                bin_acc += sp
                if rec.want_balance:
                    # баланс E-I по ПРОВОДИМОСТЯМ на RS (как в in vivo оценках g_i/g_e, Destexhe et al. 2003);
                    # по токам сравнивать нельзя: у ГАМК-А движущая сила V - E_GABA всего ~10 мВ у покоя
                    bal_e += float((self.g_ampa + gn)[self.is_exc].mean())
                    bal_i += float(self.g_gaba[self.is_exc].mean())
                if bin_steps and (self.step_i % bin_steps == 0):
                    rec.bins_t.append(self.t)
                    rec.bin_counts.append([bin_acc[self.is_exc].sum(), bin_acc[self.is_inh].sum()])
                    if rec.want_mea:
                        rec.mea_bins.append(self.mea.A @ bin_acc)
                    if rec.want_balance:
                        rec.bal_t.append(self.t)
                        rec.bal_exc.append(bal_e / bin_steps)
                        rec.bal_inh.append(bal_i / bin_steps)
                        bal_e = bal_i = 0.0
                    bin_acc = np.zeros(N)
                if v_steps and (self.step_i % v_steps == 0):
                    rec.v_t.append(self.t)
                    rec.v_snap.append(nrn.V.astype(np.float32).copy())
                if rec.gate_ids and (self.step_i % rec.gate_every == 0):
                    rec.gate_t.append(self.t)
                    for k in ("V", "m", "h", "n", "p"):
                        rec.gates[k].append(getattr(nrn, k)[rec.gate_ids].copy())
                if kk_steps and (self.step_i % kk_steps == 0):
                    rec.k_t.append(self.t)
                    rec.k_snap.append(self.gel.K.mean(axis=2).copy())
        return counts

    # ------------------------------------------------------------ сводки
    def summary(self):
        E, I = self.is_exc, self.is_inh
        k_in = self.mask.sum(axis=0)
        return dict(N=self.N, n_RS=int(E.sum()), n_FS=int(I.sum()), synapses=int(self.mask.sum()),
                    in_degree_mean=round(float(k_in.mean()), 1), long_axons=int(self.long_mask.sum()),
                    conn_radius_um=round(self.gel.conn_radius, 1), electrodes=self.mea.n_el)
