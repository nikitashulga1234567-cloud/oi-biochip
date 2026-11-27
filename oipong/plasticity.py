"""Обучение: STDP -> след пригодности -> дофамин (R-STDP) + синаптическое масштабирование.

1. STDP (парное правило с локальными следами; Bi & Poo 1998; Song, Miller & Abbott 2000):
       x_i  — пресинаптический след:  dx/dt = -x/tau_plus,  x_i += 1 при спайке i
       y_j  — постсинаптический след: dy/dt = -y/tau_minus, y_j += 1 при спайке j
       спайк post j:  STDP_ij += A_plus  * x_i     (пре раньше пост -> потенциация)
       спайк pre  i:  STDP_ij -= A_minus * y_j     (пост раньше пре -> депрессия)
2. След пригодности (eligibility trace; Izhikevich 2007, Cereb. Cortex 17:2443):
       de_ij/dt = -e_ij / tau_e + STDP_ij(t)
   STDP не меняет вес сразу, а «помечает» синапс; метка гаснет за tau_e ~ 0.5–1 с.
3. Дофамин (reward-modulated STDP; Frémaux & Gerstner 2016, обзор):
       dW_ij = lr * DA * e_ij(t_награды),   DA = g_tonic * (R - R_ожид(стимул))
   R = +1 за верный ответ, -1 за неверный; R_ожид — «критик», скользящее среднее награды для этого стимула
   (Frémaux, Sprekeler & Gerstner 2010): без вычитания ожидания R-STDP дрейфует (та же ошибка, что в v2.1).
   g_tonic — тонический дофамин (BioEnvironment.plasticity_gain из v3 или 1.0).
4. Синаптическое масштабирование (гомеостаз; Turrigiano et al. 1998, Nature 391:892):
       W_kj <- W_kj * (1 + eta * (r_target - r_j) / r_target)    для всех ВОЗБУЖДАЮЩИХ входов k нейрона j
   r_j — частота нейрона, усреднённая по пробам. Тихий нейрон усиливает входы (защита от тишины),
   разогнанный ослабляет (защита от эпилептического разгона). Мультипликативно: соотношения весов,
   выученные R-STDP, сохраняются.

Реализация следа без умножения матрицы N x N на каждом шаге: награда приходит в известный момент
t_R, поэтому вклад события в момент t сразу пишется с весом exp(-(t_R - t)/tau_e) — это точное значение
e(t_R). После выдачи награды весь след умножается на exp(-(t_R_next - t_R)/tau_e).
"""
import numpy as np


class RSTDP:
    def __init__(self, net, cfg):
        self.net, self.cfg = net, cfg
        N = net.N
        self.x = np.zeros(N)
        self.y = np.zeros(N)
        self.e = np.zeros((N, N))
        self.P = net.plastic                 # маска пластичных синапсов (E->E)
        self.Pf = self.P.astype(float)
        self.exc = net.is_exc
        self.dx = np.exp(-cfg.dt / cfg.tau_plus)
        self.dy = np.exp(-cfg.dt / cfg.tau_minus)
        self.t_reward = cfg.t_pre + cfg.t_stim + cfg.decision_lag
        self.critic = {}
        self.rate_avg = np.where(net.is_exc, cfg.rate_target_rs, cfg.rate_target_fs).astype(float)
        self.history = []

    def decay_traces(self):
        self.x *= self.dx
        self.y *= self.dy

    def on_spikes(self, sp, t):
        c = self.cfg
        w = np.exp(-(self.t_reward - t) / c.tau_elig) if t <= self.t_reward else 1.0
        ids = np.nonzero(sp)[0]
        # постсинаптические спайки: потенциация от недавних пре
        self.e[:, ids] += (c.A_plus * w) * self.x[:, None] * self.Pf[:, ids]
        # пресинаптические спайки (только RS): депрессия от недавних пост
        pre = ids[self.exc[ids]]
        if len(pre):
            self.e[pre, :] -= (c.A_minus * w) * self.y[None, :] * self.Pf[pre, :]
        self.x[ids] += 1.0
        self.y[ids] += 1.0

    def reward(self, R, stim_key, tonic_gain=1.0, next_reward_in=None):
        """Выдать награду R за пробу со стимулом stim_key. Возвращает фазический дофамин DA."""
        c, net = self.cfg, self.net
        expected = self.critic.get(stim_key, 0.0)
        DA = c.da_gain * tonic_gain * (R - expected)
        self.critic[stim_key] = expected + c.critic_rate * (R - expected)
        self.n_rewards = getattr(self, "n_rewards", 0) + 1
        # затухание скорости обучения (опция): lr_eff = lr / (1 + n / lr_decay_trials); 0 = без затухания
        lr = c.lr / (1.0 + self.n_rewards / c.lr_decay_trials) if c.lr_decay_trials else c.lr
        if DA != 0.0:
            net.W += lr * DA * self.e * self.Pf
            np.clip(net.W, 0.0, None, out=net.W)
            Wee = np.where(self.P, net.W, 0.0)
            over = Wee > c.w_max_ee
            net.W[over] = c.w_max_ee
        if next_reward_in is not None:
            self.e *= np.exp(-next_reward_in / c.tau_elig)
            self.t_reward += next_reward_in
        self.history.append(dict(R=R, expected=expected, DA=DA))
        return DA

    def scale(self, counts, dur_ms):
        """Синаптическое масштабирование после пробы: counts — спайки нейронов за пробу длиной dur_ms."""
        c, net = self.cfg, self.net
        rate = counts / (dur_ms / 1000.0)
        a = 1.0 / c.scale_tau_trials
        self.rate_avg = (1 - a) * self.rate_avg + a * rate
        if not c.scaling:
            return np.ones(net.N)
        target = np.where(net.is_exc, c.rate_target_rs, c.rate_target_fs)
        f = 1.0 + c.scale_rate * (target - self.rate_avg) / target
        f = np.clip(f, 1.0 - c.scale_clip, 1.0 + c.scale_clip)
        net.W[self.exc, :] *= f[None, :]
        np.minimum(net.W, np.where(self.P, c.w_max_ee, np.inf), out=net.W)
        return f
