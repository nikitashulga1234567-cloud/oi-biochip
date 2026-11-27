"""Измерения: базовые линии, оракул, бенчмарк по seed, абляции, ступенчатая отладка."""
import numpy as np
import pandas as pd
from .config import make_config, ROLE_SENS
from .sim import OIPong


def baseline_winrate(cfg, policy="random", n=2000, seed=0):
    """Win rate БЕЗ сети: 'random' | 'stay' | 'perfect'. Физика берётся из cfg."""
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n):
        by, vy = rng.uniform(0.1, 0.9), rng.choice([-1, 1]) * rng.uniform(0.0, 0.04)
        py, bx = rng.uniform(0.2, 0.8), 0.0
        while True:
            mv = cfg.paddle_gain * rng.integers(-3, 4) if policy == "random" else (by - py if policy == "perfect" else 0.0)
            py = float(np.clip(py + np.clip(mv, -cfg.paddle_max, cfg.paddle_max), 0.0, 1.0))
            bx += cfg.ball_vx
            by += vy
            if by < 0.0 or by > 1.0:
                vy *= -1.0
                by = float(np.clip(by, 0.0, 1.0))
            if bx >= 1.0:
                hits += abs(by - py) <= cfg.hit_half
                break
    return hits / n


def make_oracle(cfg, w=None):
    """Сеть с вручную выставленными весами (сенсор pref>0 -> UP, pref<0 -> DOWN): потолок считывания."""
    g = OIPong(cfg)
    w = cfg.w_max if w is None else w
    g.mask[:] = False
    g.W[:] = 0.0
    for s in g.idx(ROLE_SENS):
        p = g.pref[s]
        if abs(p) < 0.05:
            continue
        tgt = g.mot_up if p > 0 else g.mot_down
        g.mask[s, tgt] = True
        g.W[s, tgt] = w
    return g


def weight_structure(g):
    """Средний вес «верных» (pref>0 -> UP, pref<0 -> DOWN) и «неверных» прямых путей. Должны расходиться."""
    S = g.idx(ROLE_SENS)
    up, dn = g.mot_up, g.mot_down
    pos, neg = S[g.pref[S] > 0], S[g.pref[S] < 0]
    ok = (g.W[np.ix_(pos, up)].mean() + g.W[np.ix_(neg, dn)].mean()) / 2
    bad = (g.W[np.ix_(pos, dn)].mean() + g.W[np.ix_(neg, up)].mean()) / 2
    return float(ok), float(bad)


def benchmark(seeds=(1, 2, 3), episodes=1500, eval_every=300, eval_n=100, preset="direct", verbose=True, **cfg_kw):
    rows = []
    for sd in seeds:
        g = OIPong(make_config(preset, seed=sd, **cfg_kw))
        _, _, ev, _ = g.train(target=None, max_episodes=episodes, eval_every=eval_every, eval_n=eval_n,
                              frame_log_every=0, verbose=False)
        for _, r in ev.iterrows():
            rows.append(dict(Seed=sd, Episode=int(r["Episode"]), Eval_Win_Pct=float(r["Eval_Win_Pct"]),
                             Neurons_Total=int(r["Neurons_Total"])))
        if verbose:
            print(f"seed {sd}: " + " ".join(f"{int(r.Episode)}:{r.Eval_Win_Pct:.0f}%" for r in ev.itertuples()))
    df = pd.DataFrame(rows)
    return df, df.groupby("Episode")["Eval_Win_Pct"].agg(["mean", "std", "min", "max"]).round(1)


def ablation(variants, seeds=(1, 2, 3), episodes=1500, eval_every=300, eval_n=100, verbose=True):
    """variants: {имя: dict(preset=..., параметры Config...)}. Возвращает mean eval по вариантам."""
    out = {}
    for name, kw in variants.items():
        kw = dict(kw)
        preset = kw.pop("preset", "direct")
        if verbose:
            print(f"== {name}")
        _, summ = benchmark(seeds=seeds, episodes=episodes, eval_every=eval_every, eval_n=eval_n, preset=preset,
                            verbose=verbose, **kw)
        out[name] = summ["mean"]
    return pd.DataFrame(out)


def debug_ladder(episodes=300, eval_n=100, seed=1, verbose=True):
    """Пошаговая отладка: каждый следующий шаг добавляет одну сложность. Читать сверху вниз; шаг, на котором
    обучение ломается, показывает виновника.
       1 оракул (потолок считывания)  2 direct + обучение  3 + скрытый слой  4 + нейрогенез
       5 + температура/химия при норме  6 нагрев/охлаждение (скорость дофамина мерьте через dopamine_scan)"""
    steps = [
        ("1 оракул (потолок)", None, "direct", {}),
        ("2 direct, обучение (среда ВЫКЛ)", "learn", "direct", dict(env_enabled=False)),
        ("3 + скрытый слой", "learn", "hidden", {}),
        ("4 + нейрогенез", "learn", "hidden_structural", {}),
        ("5 + среда ВКЛ (37 °C, норма)", "learn", "direct", dict(env_enabled=True)),
        ("6a T = 32 °C", "learn", "direct", dict(temperature=32.0)),
        ("6b T = 41 °C", "learn", "direct", dict(temperature=41.0)),
    ]
    rows = []
    for name, mode, preset, kw in steps:
        cfg = make_config(preset, seed=seed, **kw)
        if mode is None:
            g = make_oracle(cfg)
            win = g.evaluate(eval_n)["win"] * 100
            rows.append(dict(Шаг=name, До=None, После=round(win, 1), Верные_W=None, Неверные_W=None))
        else:
            g = OIPong(cfg)
            before = g.evaluate(eval_n)["win"] * 100
            g.train(target=None, max_episodes=episodes, eval_every=0, frame_log_every=0, verbose=False)
            after = g.evaluate(eval_n)["win"] * 100
            ok, bad = weight_structure(g) if len(g.idx(ROLE_SENS)) and g.mask[np.ix_(g.idx(ROLE_SENS), g.mot_up)].any() else (None, None)
            rows.append(dict(Шаг=name, До=round(before, 1), После=round(after, 1),
                             Верные_W=None if ok is None else round(ok, 3), Неверные_W=None if bad is None else round(bad, 3)))
        if verbose:
            print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def temperature_scan(temps=(25.0, 30.0, 34.0, 37.0, 40.0, 42.0), n=60, seed=1):
    """Как температура меняет СЕТЬ (Q10): частота спайков сенсоров/моторов при фиксированном входе и
    win rate оракула. Чем выше T, тем быстрее кинетика каналов -> больше спайков за кадр."""
    rows = []
    for T in temps:
        cfg = make_config("direct", seed=seed, temperature=T)
        g = make_oracle(cfg)
        sens, mot = [], []
        for _ in range(30):
            g.game.reset(g.rng)
            g.game.ball_y, g.game.paddle_y = 0.8, 0.3      # большая положительная ошибка
            sp = g.step(learn=False)["spikes"]
            sens.append(sp[g.idx(ROLE_SENS)].sum())
            mot.append(sp[g.mot_up].sum())
        rows.append(dict(T_C=T, kin_factor=round(g.env.kin_factor, 3), Spikes_Sens=round(float(np.mean(sens)), 2),
                         Spikes_MotUP=round(float(np.mean(mot)), 2), Oracle_Win_Pct=round(100 * g.evaluate(n)["win"], 1)))
    return pd.DataFrame(rows)


def dopamine_scan(levels=(1.0, 0.3, 0.1, 0.0), episodes=60, eval_n=60, seed=1, preset="direct"):
    """Скорость обучения при разном тоническом дофамине (ранний eval, а не итоговый: итог часто 100 % у всех).
    При da_tonic=0 пластичность выключена (множитель lr = 0): сеть должна остаться на уровне необученной."""
    rows = []
    for lv in levels:
        g = OIPong(make_config(preset, seed=seed, da_tonic=lv, da_release=0.0 if lv == 0 else 0.15))
        before = g.evaluate(eval_n)["win"] * 100
        g.train(target=None, max_episodes=episodes, eval_every=0, frame_log_every=0, verbose=False)
        ok, bad = weight_structure(g)
        rows.append(dict(DA_tonic=lv, Gain_lr=round(g.env.plasticity_gain, 3), Eval_До=round(before, 1),
                         Eval_После=round(g.evaluate(eval_n)["win"] * 100, 1), Верные_W=round(ok, 3), Неверные_W=round(bad, 3)))
    return pd.DataFrame(rows)
