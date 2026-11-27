"""Нейрон уровня 2: проводимостная модель Pospischil et al. (2008).

Источник: Pospischil M., Toledo-Rodriguez M., Monier C., Piwkowska Z., Bal T., Frégnac Y., Markram H.,
Destexhe A. «Minimal Hodgkin–Huxley type models for different classes of cortical and thalamic neurons».
Biol. Cybern. 99:427–441 (2008). Кинетика каналов в статье дана для температуры 36 °C.

Одно-компартментный нейрон (цилиндр L = d), уравнение мембраны (ур. 1 статьи):

    C_m dV/dt = - g_L (V - E_L) - I_Na - I_Kd - I_M - I_syn + I_ext

    I_Na = g_Na m^3 h (V - E_Na)        быстрый натрий, Traub & Miles 1991 (ур. 4)
    I_Kd = g_Kd n^4   (V - E_K)         задержанный выпрямитель калия (ур. 5)
    I_M  = g_M  p     (V - E_K)         медленный K+ (M-ток), Yamada et al. 1989: адаптация частоты у RS

Скорости (1/мс), V в мВ, V_T сдвигает порог спайка:
    a_m = -0.32 (V-VT-13) / (exp(-(V-VT-13)/4) - 1)      b_m = 0.28 (V-VT-40) / (exp((V-VT-40)/5) - 1)
    a_h = 0.128 exp(-(V-VT-17)/18)                        b_h = 4 / (1 + exp(-(V-VT-40)/5))
    a_n = -0.032 (V-VT-15) / (exp(-(V-VT-15)/5) - 1)      b_n = 0.5 exp(-(V-VT-10)/40)
    p_inf = 1 / (1 + exp(-(V+35)/10))
    tau_p = tau_max / (3.3 exp((V+35)/20) + exp(-(V+35)/20))

Единицы: C_m = 1 мкФ/см2, проводимости в мСм/см2, токи в мкА/см2 (=> dV/dt в мВ/мс), время в мс.
Пересчёт в абсолютные единицы: площадь цилиндра L = d равна pi*d*L (торцы в статье не учитываются):
RS d = 61.4 мкм -> 1.18e-4 см2, 1 мкА/см2 = 0.118 нА.

Параметры клеток (взяты из подписей к рисункам статьи, сверены с PDF):
  RS (возбуждающая пирамидная, Fig. 2a, соматосенсорная кора крысы): L = d = 61.4 мкм,
      g_L = 2.05e-5 См/см2 (0.0205 мСм/см2), E_L = -70.3 мВ, g_Na = 0.056 См/см2, V_T = -56.2 мВ,
      g_Kd = 0.006 См/см2, g_M = 7.5e-5 См/см2, tau_max = 608 мс.
  FS (быстрая тормозная, Fig. 4, та же кора): L = d = 56.9 мкм, g_L = 3.8e-5 См/см2, E_L = -70.4 мВ,
      g_Na = 0.058 См/см2, V_T = -57.9 мВ, g_Kd = 0.0039 См/см2.
      В Fig. 4 у подогнанной FS-клетки есть слабый M-ток (7.87e-5 См/см2, tau_max 502 мс); по ТЗ и по модели
      FS из Fig. 3 статьи («no I_M») мы его ОБНУЛЯЕМ: FS без адаптации. Включить: fs_gM=0.0787.
  E_Na = 50 мВ, E_K = -90 мВ (раздел 2.2 статьи).

Интегрирование: экспоненциальный метод Эйлера (Rush–Larsen для затворов и для V при замороженных
проводимостях): x(t+dt) = x_inf + (x - x_inf) * exp(-dt / tau_x). Он безусловно устойчив для линейного по x
уравнения; шаг 0.05 мс (точность проверяется сравнением с 0.025 мс в run_checks.py).

Температура: скорости затворов умножаются на phi = Q10_gate ** ((T - 36) / 10) (см. gel.py).
"""
import numpy as np

E_NA, E_K = 50.0, -90.0      # мВ, Pospischil 2008, раздел 2.2
T_REF_KIN = 36.0             # °C, температура, к которой относится кинетика (Pospischil 2008, раздел 2.2)

# Параметры типов (мСм/см2, мВ, мс, мкм). Индекс типа: 0 = RS, 1 = FS.
RS, FS = 0, 1
TYPE_NAME = {RS: "RS", FS: "FS"}
CELL_PARAMS = {
    RS: dict(L=61.4, gL=0.0205, EL=-70.3, gNa=56.0, VT=-56.2, gKd=6.0, gM=0.075, tau_max=608.0),
    FS: dict(L=56.9, gL=0.038, EL=-70.4, gNa=58.0, VT=-57.9, gKd=3.9, gM=0.0, tau_max=502.0),
}


def area_cm2(L_um):
    """Площадь боковой поверхности цилиндра L = d, см2 (1 мкм2 = 1e-8 см2)."""
    return np.pi * L_um * L_um * 1e-8


def _vtrap(x, y):
    """x / (exp(x/y) - 1) без деления 0/0 при x -> 0 (предел равен y)."""
    x = np.asarray(x, dtype=float)
    small = np.abs(x / y) < 1e-6
    xs = np.where(small, 1.0, x)
    return np.where(small, y * (1.0 - x / y / 2.0), xs / np.expm1(xs / y))


def rates_u(u):
    """Скорости Na и Kd как функция u = V - V_T (мВ). Возвращает (a_m, b_m, a_h, b_h, a_n, b_n), 1/мс."""
    a_m = 0.32 * _vtrap(-(u - 13.0), 4.0)          # = -0.32 (u-13) / (exp(-(u-13)/4) - 1)
    b_m = 0.28 * _vtrap(u - 40.0, 5.0)             # =  0.28 (u-40) / (exp((u-40)/5) - 1)
    a_h = 0.128 * np.exp(-(u - 17.0) / 18.0)
    b_h = 4.0 / (1.0 + np.exp(-(u - 40.0) / 5.0))
    a_n = 0.032 * _vtrap(-(u - 15.0), 5.0)
    b_n = 0.5 * np.exp(-(u - 10.0) / 40.0)
    return a_m, b_m, a_h, b_h, a_n, b_n


def m_current_kin(V, tau_max):
    """M-ток: p_inf(V) и tau_p(V) (мс), Yamada et al. 1989 в редакции Pospischil 2008."""
    p_inf = 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
    tau_p = tau_max / (3.3 * np.exp((V + 35.0) / 20.0) + np.exp(-(V + 35.0) / 20.0))
    return p_inf, tau_p


class GateTables:
    """Таблицы x_inf и exp(-dt*phi/tau) на сетке напряжения (шаг 0.02 мВ): ускорение в ~5 раз.
    Na/Kd табулируются по u = V - V_T (общая таблица для RS и FS), M-ток по V для каждого tau_max."""
    LO, HI, RES = -160.0, 120.0, 50.0       # мВ, точек на мВ

    def __init__(self, dt, phi, tau_maxes):
        self.dt, self.phi = dt, phi
        g = np.arange(self.LO, self.HI + 1.0 / self.RES, 1.0 / self.RES)
        self.grid = g
        a_m, b_m, a_h, b_h, a_n, b_n = rates_u(g)
        k = dt * phi
        self.m_inf, self.m_e = a_m / (a_m + b_m), np.exp(-k * (a_m + b_m))
        self.h_inf, self.h_e = a_h / (a_h + b_h), np.exp(-k * (a_h + b_h))
        self.n_inf, self.n_e = a_n / (a_n + b_n), np.exp(-k * (a_n + b_n))
        self.p = {}
        for tm in set(tau_maxes):
            p_inf, tau_p = m_current_kin(g, tm)
            self.p[tm] = (p_inf, np.exp(-k / tau_p))
        self.top = len(g) - 1

    def index(self, x):
        return np.clip(((x - self.LO) * self.RES + 0.5).astype(np.int64), 0, self.top)


def steady_state(V, VT, tau_max=608.0):
    """Стационарные значения затворов при напряжении V: (m, h, n, p)."""
    a_m, b_m, a_h, b_h, a_n, b_n = rates_u(np.asarray(V) - VT)
    p_inf, _ = m_current_kin(np.asarray(V), tau_max)
    return a_m / (a_m + b_m), a_h / (a_h + b_h), a_n / (a_n + b_n), p_inf


class NeuronArray:
    """Массив одно-компартментных нейронов Pospischil (векторно по numpy).
    Состояние: V, m, h, n, p. Параметры — по типу (RS/FS) в массивах длины N."""

    def __init__(self, types, dt=0.05, phi=1.0, cond_factor=1.0, fs_gM=0.0, V0=None):
        self.types = np.asarray(types, dtype=int)
        N = len(self.types)
        self.N, self.dt = N, dt
        P = {k: np.array([CELL_PARAMS[t][k] for t in self.types], dtype=float) for k in CELL_PARAMS[RS]}
        if fs_gM:
            P["gM"][self.types == FS] = fs_gM
        self.P = P
        self.area = area_cm2(P["L"])
        self.set_temperature_factors(phi, cond_factor)
        V0 = np.where(self.types == RS, CELL_PARAMS[RS]["EL"], CELL_PARAMS[FS]["EL"]) if V0 is None else V0
        self.V = np.array(V0, dtype=float) * np.ones(N)
        m, h, n, p = steady_state(self.V, P["VT"], 608.0)
        self.m, self.h, self.n = m.copy(), h.copy(), n.copy()
        self.p = np.array([m_current_kin(v, tm)[0] for v, tm in zip(self.V, P["tau_max"])])
        self.above = np.zeros(N, dtype=bool)
        self.EK_shift = np.zeros(N)      # сдвиг E_K от внеклеточного K+ (gel.py), мВ

    def set_temperature_factors(self, phi, cond_factor=1.0):
        """phi: множитель скоростей затворов (Q10), cond_factor: множитель максимальных проводимостей."""
        self.phi, self.cond_factor = phi, cond_factor
        self.tab = GateTables(self.dt, phi, self.P["tau_max"])
        self._tm_groups = [(tm, np.where(self.P["tau_max"] == tm)[0]) for tm in sorted(set(self.P["tau_max"]))]
        c = cond_factor
        self.gNa, self.gKd, self.gM, self.gL = c * self.P["gNa"], c * self.P["gKd"], c * self.P["gM"], c * self.P["gL"]

    def step(self, I_ext, g_exc=None, g_nmda_eff=None, g_inh=None, E_inh=-75.0):
        """Один шаг dt. I_ext: мкА/см2 (вход положительный = деполяризация).
        g_exc (AMPA, E=0), g_nmda_eff (NMDA с уже умноженным блоком Mg, E=0), g_inh (ГАМК-А, E=E_inh): мСм/см2.
        Возвращает булев массив «спайк на этом шаге» (пересечение 0 мВ снизу вверх)."""
        tab, V = self.tab, self.V
        iu = tab.index(V - self.P["VT"])
        self.m = tab.m_inf[iu] + (self.m - tab.m_inf[iu]) * tab.m_e[iu]
        self.h = tab.h_inf[iu] + (self.h - tab.h_inf[iu]) * tab.h_e[iu]
        self.n = tab.n_inf[iu] + (self.n - tab.n_inf[iu]) * tab.n_e[iu]
        iv = tab.index(V)
        for tm, idx in self._tm_groups:
            p_inf, p_e = tab.p[tm]
            j = iv[idx]
            self.p[idx] = p_inf[j] + (self.p[idx] - p_inf[j]) * p_e[j]
        EK = E_K + self.EK_shift
        gna = self.gNa * self.m ** 3 * self.h
        gk = self.gKd * self.n ** 4 + self.gM * self.p
        G = gna + gk + self.gL                      # полная проводимость, мСм/см2
        GE = gna * E_NA + gk * EK + self.gL * self.P["EL"]
        if g_exc is not None:
            G = G + g_exc                           # E_AMPA = 0 -> вклад в GE нулевой
        if g_nmda_eff is not None:
            G = G + g_nmda_eff
        if g_inh is not None:
            G = G + g_inh
            GE = GE + g_inh * E_inh
        V_inf = (GE + I_ext) / G                    # мВ; C_m = 1 мкФ/см2 -> tau_V = 1/G мс
        self.V = V_inf + (V - V_inf) * np.exp(-self.dt * G)
        now = self.V > 0.0
        sp = now & ~self.above
        self.above = now
        return sp


def simulate_single(cell_type, I_uA_cm2, t_ms=1000.0, dt=0.05, phi=1.0, delay_ms=100.0, record=False, fs_gM=0.0):
    """Одиночный нейрон, ступенька тока I (мкА/см2) с delay_ms до t_ms. Возвращает времена спайков (мс)
    и, если record, словарь трасс V, m, h, n, p (каждый 4-й шаг)."""
    nrn = NeuronArray([cell_type], dt=dt, phi=phi, fs_gM=fs_gM)
    steps = int(round(t_ms / dt))
    spikes, rec = [], {k: [] for k in ("t", "V", "m", "h", "n", "p")}
    for s in range(steps):
        t = s * dt
        I = I_uA_cm2 if t >= delay_ms else 0.0
        if nrn.step(np.array([I]))[0]:
            spikes.append(t)
        if record and s % 4 == 0:
            rec["t"].append(t)
            for k in ("V", "m", "h", "n", "p"):
                rec[k].append(float(getattr(nrn, k)[0]))
    return np.array(spikes), ({k: np.array(v) for k, v in rec.items()} if record else None)


def fi_curve(cell_type, currents, t_ms=1100.0, delay_ms=100.0, dt=0.05, phi=1.0):
    """f–I кривая: для каждого тока — средняя частота за 1 с стимула, частота первых двух спайков
    (1/ISI1) и последних (1/ISI_last) и индекс адаптации ISI_last/ISI_first."""
    rows = []
    for I in currents:
        sp, _ = simulate_single(cell_type, I, t_ms=t_ms, dt=dt, phi=phi, delay_ms=delay_ms)
        sp = sp[sp >= delay_ms]
        isi = np.diff(sp)
        rows.append(dict(I_uA_cm2=float(I), I_nA=round(float(I * area_cm2(CELL_PARAMS[cell_type]["L"]) * 1e3), 4),
                         rate_Hz=len(sp) / ((t_ms - delay_ms) / 1000.0),
                         f_first_Hz=1000.0 / isi[0] if len(isi) else 0.0,
                         f_last_Hz=1000.0 / isi[-1] if len(isi) else 0.0,
                         adapt=float(isi[-1] / isi[0]) if len(isi) > 1 else np.nan))
    return rows
