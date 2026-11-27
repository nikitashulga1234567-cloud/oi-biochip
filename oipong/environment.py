"""BioEnvironment: температура и химическая среда (глутамат, ГАМК, дофамин).

Температура масштабирует кинетику через Q10:   фактор(T) = Q10 ** ((T - T_ref) / 10)
  - воротные переменные m, h, n (Q10 = 3):  dm/dt умножается на фактор (спайки быстрее при нагреве);
  - проводимости каналов (Q10 ~ 1.4):        g_Na, g_K, g_L умножаются на фактор;
  - синапс (Q10 ~ 2):                         время спада tau делится на фактор.
Параметры HH считаются откалиброванными при T_ref (по умолчанию 37 °C). Классические HH-параметры
получены при 6.3 °C; чтобы их использовать «как есть», поставьте temp_ref=6.3.

Нейромедиаторы: относительные концентрации (1.0 = референсный фон). Каждый пул:
    C(t+dt) = tonic + (C - tonic) * exp(-dt / tau) + выброс
Выброс: глутамат от возбуждающих спайков, ГАМК от тормозных, фазический дофамин от |награды|.
Влияние (функция Хилла, нормированная так, что при C = 1 множитель равен 1, при C = 0 равен 0):
    gain(C) = 2*C / (C + 1)      (при env_k = 1; ограничена сверху значением 2)
  - глутамат -> усиление возбуждающих синапсов;  ГАМК -> усиление тормозных;
  - дофамин  -> множитель скорости пластичности lr (при 0 обучение выключено).
Пространственной диффузии нет: пулы «хорошо перемешаны» (расширение: сетка вокселей).
"""
import numpy as np


class BioEnvironment:
    def __init__(self, cfg):
        self.cfg = cfg
        self.T = float(cfg.temperature)
        self.tonic = dict(glu=cfg.glu_tonic, gaba=cfg.gaba_tonic, da=cfg.da_tonic)
        self.level = dict(self.tonic)
        self.tau = dict(glu=cfg.glu_tau, gaba=cfg.gaba_tau, da=cfg.da_tau)
        self.history = []

    # ---------------- температура
    def _q(self, q10):
        return float(q10 ** ((self.T - self.cfg.temp_ref) / 10.0))

    @property
    def kin_factor(self):      # воротные переменные
        return self._q(self.cfg.q10_gate) if self.cfg.env_enabled else 1.0

    @property
    def cond_factor(self):     # проводимости
        return self._q(self.cfg.q10_cond) if self.cfg.env_enabled else 1.0

    @property
    def syn_rate_factor(self):  # скорость спада синапса
        return self._q(self.cfg.q10_syn) if self.cfg.env_enabled else 1.0

    # ---------------- химия
    def _gain(self, c):
        k = self.cfg.env_k
        ref = 1.0 / (1.0 + k)
        return (max(c, 0.0) / (max(c, 0.0) + k)) / ref if self.cfg.env_enabled else 1.0

    @property
    def glu_gain(self):
        return self._gain(self.level["glu"])

    @property
    def gaba_gain(self):
        return self._gain(self.level["gaba"])

    @property
    def plasticity_gain(self):
        if not (self.cfg.env_enabled and self.cfg.da_gate_plasticity):
            return 1.0
        return self._gain(self.level["da"])

    def set_temperature(self, T):
        self.T = float(T)

    def set_tonic(self, name, value):
        """Например env.set_tonic('da', 0.2): истощение дофамина (обучение почти выключено)."""
        self.tonic[name] = float(value)

    def step(self, dt_ms, n_exc_spikes=0.0, n_inh_spikes=0.0, reward=0.0):
        """Обновить пулы за интервал dt_ms (обычно один кадр)."""
        c = self.cfg
        for k in self.level:
            self.level[k] = self.tonic[k] + (self.level[k] - self.tonic[k]) * np.exp(-dt_ms / self.tau[k])
        self.level["glu"] += c.glu_release * n_exc_spikes
        self.level["gaba"] += c.gaba_release * n_inh_spikes
        self.level["da"] += c.da_release * abs(reward)

    def snapshot(self):
        return dict(T=float(self.T), kin=round(float(self.kin_factor), 4), Glu=round(float(self.level["glu"]), 4),
                    GABA=round(float(self.level["gaba"]), 4), DA=round(float(self.level["da"]), 4),
                    plasticity_gain=round(float(self.plasticity_gain), 4))

    # для сохранения
    def state(self):
        return dict(T=self.T, tonic=dict(self.tonic), level=dict(self.level))

    def load(self, d):
        self.T, self.tonic, self.level = d["T"], dict(d["tonic"]), dict(d["level"])
