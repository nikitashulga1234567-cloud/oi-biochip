"""Задача дискриминации стимулов A/B на HD-MEA (по мотивам DishBrain: стимул -> ответ -> обратная связь).

Проба (длительности в BioConfig):
    t_pre   пауза без стимула
    t_stim  стимул A или B: гауссово поле деполяризации от кластера электродов (mea.py), центр и амплитуда
            с разбросом от пробы к пробе (stim_jitter_pos, stim_jitter_amp) — чтобы сеть не заучила один вход
    lag     хвост окна решения
    -> РЕШЕНИЕ: сигнал «моторных» кластеров электродов out_1 и out_2 (считывание MEA с затуханием) за окно
       [начало стимула, конец стимула + lag]; ответ = кластер с большим сигналом
    -> НАГРАДА: +1 за верный ответ, -1 за неверный -> фазический дофамин -> R-STDP (plasticity.py)
    t_post  пауза после награды (след пригодности гаснет, сеть возвращается к фону)
    -> синаптическое масштабирование по частотам за пробу

Правильный ответ (mapping='straight'): A -> out_1, B -> out_2. Каждый выходной кластер равноудалён от A и B,
поэтому геометрия ответа не подсказывает: необученная сеть около 50 %, рост — заслуга обучения.
(В пробном прогоне с асимметричной геометрией ответ был детерминирован расстоянием, и R-STDP с критиком
не учился: при всегда одинаковом исходе R - R_ожид -> 0. Для обучения с подкреплением нужен разброс исходов.)

Оценка: отдельный ГСЧ (другой seed), без пластичности и без масштабирования, новые пробы — «отложенная» выборка.
"""
import time
import numpy as np
from .bionet import BioNet, Recorder
from .plasticity import RSTDP


def dprime(x_a, x_b):
    """Индекс разделимости d' = (mu_A - mu_B) / sqrt((var_A + var_B) / 2)."""
    x_a, x_b = np.asarray(x_a, float), np.asarray(x_b, float)
    if len(x_a) < 2 or len(x_b) < 2:
        return float("nan")
    s = np.sqrt((x_a.var(ddof=1) + x_b.var(ddof=1)) / 2.0)
    return float((x_a.mean() - x_b.mean()) / s) if s > 0 else float("nan")


class ABTask:
    def __init__(self, cfg, net=None, env=None):
        self.cfg = cfg
        self.net = net or BioNet(cfg)
        self.env = env                                  # BioEnvironment v3 (тонический дофамин) или None
        self.rule = RSTDP(self.net, cfg)
        self.net.plasticity = self.rule
        m = self.net.mea
        self.el_out = [m.electrodes_near(cfg.out_1, cfg.cluster_radius), m.electrodes_near(cfg.out_2, cfg.cluster_radius)]
        self.el_in = [m.electrodes_near(cfg.stim_A, cfg.cluster_radius), m.electrodes_near(cfg.stim_B, cfg.cluster_radius)]
        self.read_out = np.stack([m.A[e].sum(axis=0) for e in self.el_out])      # (2, N): вклад нейронов в кластер
        self.read_in = np.stack([m.A[e].sum(axis=0) for e in self.el_in])
        self.target = {"A": 1, "B": 0} if cfg.mapping == "crossed" else {"A": 0, "B": 1}
        self.log = []
        self.trial_i = 0
        self.margin_var = 100.0                         # начальная оценка дисперсии запаса решения
        self.sig_base = np.zeros(2)                     # скользящее среднее сигнала out1/out2 (center_decision)
        self.period = cfg.t_pre + cfg.t_stim + cfg.decision_lag + cfg.t_post
        self.net.run(200.0)          # прогрев: сеть уходит из покоя в фоновый режим

    def stim_vector(self, label, rng):
        c = self.cfg
        cx, cy = c.stim_A if label == "A" else c.stim_B
        cx, cy = cx + rng.normal(0, c.stim_jitter_pos), cy + rng.normal(0, c.stim_jitter_pos)
        amp = c.stim_amp * (1.0 + rng.uniform(-c.stim_jitter_amp, c.stim_jitter_amp))
        return self.net.mea.stim_current([(cx, cy)], amp)

    def trial(self, label, learn=True, rng=None, rec=None):
        """Одна проба. Возвращает словарь: метка, ответ, верно ли, сигналы кластеров, дофамин."""
        c, net = self.cfg, self.net
        rng = rng or net.rng
        I = self.stim_vector(label, rng)
        total = np.zeros(net.N)
        t0 = net.t
        if learn:   # момент награды этой пробы известен заранее: след пригодности пишется относительно него
            t_r = net.t + c.t_pre + c.t_stim + c.decision_lag
            if abs(self.rule.t_reward - t_r) > 1e-6:
                self.rule.e *= np.exp(-max(t_r - self.rule.t_reward, 0.0) / c.tau_elig)
                self.rule.t_reward = t_r
        total += net.run(c.t_pre, None, rec, learn)
        if rec is not None:
            rec.stim_label.append((net.t, net.t + c.t_stim, label))
        dec = net.run(c.t_stim, I, rec, learn)
        dec += net.run(c.decision_lag, None, rec, learn)
        total += dec
        raw = self.read_out @ dec + rng.normal(0, c.mea_noise * np.sqrt(len(self.el_out[0])), 2)
        sig_in = self.read_in @ dec
        # центрирование решения: вычитаем скользящее среднее сигнала каждого кластера по пробам (обе метки).
        # Аналог post - <post> в RM-Hebb v3: убирает общий дрейф «out1 сильнее out2», оставляя стимул-специфичную
        # часть. Среднее обновляется только при обучении; в оценке порог заморожен.
        sig = raw - self.sig_base if c.center_decision else raw
        if learn and c.center_decision:
            self.sig_base += c.center_rate * (raw - self.sig_base)
        if sig[0] == sig[1]:
            choice = int(rng.integers(2))
        else:
            choice = int(np.argmax(sig))
        correct = choice == self.target[label]
        # запас решения: сигнал «верного» кластера минус «неверного» (>0 — верно)
        margin = float(sig[self.target[label]] - sig[1 - self.target[label]])
        if c.reward_mode == "margin":
            # градуированная награда (как shaping в Pong v3): R = tanh(запас / масштаб), масштаб = скользящее
            # стандартное отклонение запаса. Меньше дисперсия, чем у ±1: сигнал есть и в «уверенно верных» пробах.
            if learn:
                self.margin_var = 0.98 * self.margin_var + 0.02 * margin ** 2
            R = float(np.tanh(margin / max(np.sqrt(self.margin_var), 1e-9)))
        else:
            R = 1.0 if correct else -1.0
        DA = 0.0
        if learn:
            tonic = self.env.plasticity_gain if self.env is not None else 1.0
            DA = self.rule.reward(R, label, tonic, next_reward_in=self.period)
            if self.env is not None:
                self.env.step(c.t_stim, reward=DA)
        total += net.run(c.t_post, None, rec, learn)
        if learn:
            self.rule.scale(total, net.t - t0)
        rate_rs = total[net.is_exc].mean() / ((net.t - t0) / 1000.0)
        rate_fs = total[net.is_inh].mean() / ((net.t - t0) / 1000.0)
        out = dict(trial=self.trial_i, label=label, choice=choice, correct=bool(correct), R=R, DA=DA, margin=margin,
                   out1=float(raw[0]), out2=float(raw[1]), dv=float(sig[self.target["A"]] - sig[self.target["B"]]),
                   in_A=float(sig_in[0]), in_B=float(sig_in[1]), rate_RS=float(rate_rs), rate_FS=float(rate_fs),
                   T=net.gel.T, K_mean=float(net.gel.K.mean()), w_ee_mean=float(net.W[net.plastic].mean()))
        if learn:
            self.trial_i += 1
        return out

    def train(self, n_trials, verbose_every=50, eval_every=None, eval_n=60, eval_seed=1000, callback=None):
        """Обучение n_trials проб (A/B в случайном порядке). Возвращает журнал проб и оценок."""
        rng = self.net.rng
        evals = []
        t0 = time.time()
        for k in range(n_trials):
            lab = "A" if rng.random() < 0.5 else "B"
            r = self.trial(lab, learn=True)
            self.log.append(r)
            if verbose_every and (k + 1) % verbose_every == 0:
                last = self.log[-verbose_every:]
                print(f"  проба {k+1:4d}: train-точность {100*np.mean([x['correct'] for x in last]):5.1f}% | "
                      f"RS {np.mean([x['rate_RS'] for x in last]):.2f} Гц FS {np.mean([x['rate_FS'] for x in last]):.2f} Гц"
                      f" | w_EE {r['w_ee_mean']:.5f} | {time.time()-t0:.0f} с", flush=True)
            if eval_every and (k + 1) % eval_every == 0:
                ev = self.evaluate(eval_n, seed=eval_seed)
                ev["trial"] = k + 1
                evals.append(ev)
                print(f"  оценка после {k+1}: {100*ev['acc']:.1f}% (d' {ev['dprime']:.2f})", flush=True)
            if callback:
                callback(self, r)
        return self.log, evals

    def calibrate(self, n=40, seed=999):
        """Калибровка порога решения БЕЗ обучения: sig_base = средний сигнал out1/out2 за n проб.
        Делается до оценки «до обучения», чтобы и она шла с центрированным решением (честное сравнение)."""
        net = self.net
        saved = self._save_dynamic()
        rng = np.random.default_rng(seed)
        old = net.rng
        net.rng = rng
        raws = []
        try:
            for k in range(n):
                r = self.trial("A" if k % 2 == 0 else "B", learn=False, rng=rng)
                raws.append([r["out1"], r["out2"]])
        finally:
            net.rng = old
            self._load_dynamic(saved)
        self.sig_base = np.mean(raws, axis=0)
        return self.sig_base

    def evaluate(self, n=100, seed=1000, rec=None):
        """Отложенная оценка: n проб (поровну A/B) на отдельном ГСЧ, БЕЗ пластичности и масштабирования.
        Состояние сети (V, проводимости, K+, время) восстанавливается после оценки."""
        net = self.net
        saved = self._save_dynamic()
        rng = np.random.default_rng(seed)
        labels = np.array(["A", "B"] * (n // 2))
        rng.shuffle(labels)
        res = []
        try:
            net.rng, old_rng = rng, net.rng
            for lab in labels:
                res.append(self.trial(str(lab), learn=False, rng=rng, rec=rec))
        finally:
            net.rng = old_rng
            self._load_dynamic(saved)
        acc = float(np.mean([r["correct"] for r in res]))
        dv_a = [r["dv"] for r in res if r["label"] == "A"]
        dv_b = [r["dv"] for r in res if r["label"] == "B"]
        return dict(acc=acc, dprime=dprime(dv_a, dv_b), n=len(res), trials=res)

    def _save_dynamic(self):
        n, r = self.net, self.rule
        return dict(V=n.nrn.V.copy(), m=n.nrn.m.copy(), h=n.nrn.h.copy(), nn=n.nrn.n.copy(), p=n.nrn.p.copy(),
                    above=n.nrn.above.copy(), ga=n.g_ampa.copy(), gn=n.g_nmda.copy(), gg=n.g_gaba.copy(),
                    buf=n.buf.copy(), t=n.t, si=n.step_i, K=n.gel.K.copy(), x=r.x.copy(), y=r.y.copy())

    def _load_dynamic(self, d):
        n, r = self.net, self.rule
        n.nrn.V, n.nrn.m, n.nrn.h, n.nrn.n, n.nrn.p, n.nrn.above = d["V"], d["m"], d["h"], d["nn"], d["p"], d["above"]
        n.g_ampa, n.g_nmda, n.g_gaba, n.buf, n.t, n.step_i = d["ga"], d["gn"], d["gg"], d["buf"], d["t"], d["si"]
        n.gel.K, r.x, r.y = d["K"], d["x"], d["y"]


def run_ab_experiment(seed=1, n_train=400, n_eval=100, eval_seed=None, verbose=True, **cfg_kw):
    """Полный опыт для одного seed: оценка до обучения -> обучение -> оценка после (отложенная выборка).
    Возвращает словарь с точностями, d', временем расчёта и объектом задачи."""
    from .bioconfig import make_bio_config
    cfg = make_bio_config(seed=seed, **cfg_kw)
    t0 = time.time()
    task = ABTask(cfg)
    if cfg.center_decision:
        task.calibrate()
    es = 10_000 + seed if eval_seed is None else eval_seed
    before = task.evaluate(n_eval, seed=es)
    if verbose:
        print(f"[seed {seed}] до обучения: {100*before['acc']:.1f}% (d' {before['dprime']:.2f})", flush=True)
    W_before = task.net.W.copy()
    task.train(n_train, verbose_every=50 if verbose else 0)
    after = task.evaluate(n_eval, seed=es)
    el = time.time() - t0
    if verbose:
        print(f"[seed {seed}] после обучения: {100*after['acc']:.1f}% (d' {after['dprime']:.2f}), {el/60:.1f} мин",
              flush=True)
    return dict(seed=seed, N=cfg.n_neurons, acc_before=before["acc"], acc_after=after["acc"],
                dprime_before=before["dprime"], dprime_after=after["dprime"], n_train=n_train, n_eval=n_eval,
                minutes=el / 60, task=task, W_before=W_before, eval_after=after, eval_before=before)
