"""Проверочные прогоны v4 (то, что записано в README, раздел «Что проверено запуском»).

    python run_checks.py fi                         # (а) f–I одиночных RS/FS, адаптация, шаг 0.05 vs 0.025 мс
    python run_checks.py ab --seeds 1 2 3 --train 900 --eval 200   # (б) A/B при N=1000, сиды параллельно
    python run_checks.py viz --seed 1               # (в) панель V1–V6 (PNG) + анимация (GIF/MP4) + TE/MI
    python run_checks.py env --seed 1               # среда: температура, Mg2+, вязкость (на обученной сети)

Результаты: out/*.json, out/*.png, out/*.gif, out/*.mp4. Обученные веса: out/ab_seed<N>.npz.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.environ.get("OI_OUT", os.path.join(HERE, "out"))   # каталог результатов (переменная OI_OUT)
os.makedirs(OUT, exist_ok=True)


def _dump(name, obj):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=float)


# ----------------------------------------------------------------------------- (а)
def check_fi():
    from oipong.neuron import RS, FS, fi_curve, simulate_single
    t0 = time.time()
    cur = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
    res = {}
    for dt in (0.025, 0.05, 0.1):
        res[dt] = {nm: fi_curve(t, cur, dt=dt) for nm, t in (("RS", RS), ("FS", FS))}
    print("ток мкА/см2 (нА RS) | RS Гц (1/ISI первый -> последний) | FS Гц (первый -> последний) | RS/FS Гц при dt 0.025 и 0.1")
    for k, I in enumerate(cur):
        r, f = res[0.05]["RS"][k], res[0.05]["FS"][k]
        print(f"  {I:5.2f} ({r['I_nA']:.3f}) | RS {r['rate_Hz']:6.1f} ({r['f_first_Hz']:6.1f} -> {r['f_last_Hz']:6.1f}) "
              f"| FS {f['rate_Hz']:6.1f} ({f['f_first_Hz']:6.1f} -> {f['f_last_Hz']:6.1f}) "
              f"| {res[0.025]['RS'][k]['rate_Hz']:.0f}/{res[0.025]['FS'][k]['rate_Hz']:.0f}, "
              f"{res[0.1]['RS'][k]['rate_Hz']:.0f}/{res[0.1]['FS'][k]['rate_Hz']:.0f}")
    dev = max(abs(res[d][nm][k]["rate_Hz"] - res[0.025][nm][k]["rate_Hz"]) / max(res[0.025][nm][k]["rate_Hz"], 1.0)
              for d in (0.05, 0.1) for nm in ("RS", "FS") for k in range(len(cur)))
    print(f"макс. относительное расхождение частоты с dt = 0.025 мс: {100*dev:.1f}%")
    traces = {}
    for nm, t in (("RS, 1.5 мкА/см² (адаптация)", RS), ("FS, 1.5 мкА/см² (без адаптации)", FS)):
        _, tr = simulate_single(t, 1.5, t_ms=600.0, delay_ms=50.0, record=True)
        traces[nm] = tr
    try:
        from oipong import viz
        viz.fi_figure(res[0.05]["RS"], res[0.05]["FS"], traces, path=os.path.join(OUT, "fi_curves.png"))
        print("сохранено out/fi_curves.png")
    except ImportError as e:
        print("matplotlib нет, рисунок не строился:", e)
    _dump("fi.json", {str(k): v for k, v in res.items()})
    print(f"время: {time.time()-t0:.1f} с")


# ----------------------------------------------------------------------------- (б)
def _ab_worker(args):
    seed, n_train, n_eval, eval_every, kw = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from oipong.bioconfig import make_bio_config, bio_cfg_to_dict
    from oipong.task_ab import ABTask
    t0 = time.time()
    cfg = make_bio_config(seed=seed, **kw)
    task = ABTask(cfg)
    if cfg.center_decision:
        task.calibrate()                    # порог решения до обучения (без пластичности)
    es = 10_000 + seed                      # ГСЧ отложенной оценки: не пересекается с обучением
    before = task.evaluate(n_eval, seed=es)
    W_before = task.net.W.copy()
    evals = [dict(trial=0, acc=before["acc"], dprime=before["dprime"])]
    done = 0
    while done < n_train:
        k = min(eval_every, n_train - done)
        task.train(k, verbose_every=0)
        done += k
        ev = task.evaluate(n_eval, seed=es)
        evals.append(dict(trial=done, acc=ev["acc"], dprime=ev["dprime"]))
        print(f"[seed {seed}] проба {done}: отложенная оценка {100*ev['acc']:.1f}% d' {ev['dprime']:.2f} "
              f"| {(time.time()-t0)/60:.1f} мин", flush=True)
    # независимая повторная оценка на другом ГСЧ
    ev2 = task.evaluate(n_eval, seed=es + 500)
    np.savez_compressed(os.path.join(OUT, f"ab_seed{seed}.npz"), W=task.net.W, W_before=W_before,
                        sig_base=task.sig_base)
    log = [{k: v for k, v in r.items()} for r in task.log]
    _dump(f"ab_seed{seed}_log.json", dict(cfg=bio_cfg_to_dict(cfg), evals=evals, train_log=log))
    return dict(seed=seed, N=cfg.n_neurons, n_train=n_train, n_eval=n_eval, acc_before=before["acc"],
                acc_after=evals[-1]["acc"], acc_after_seed2=ev2["acc"], dprime_before=before["dprime"],
                dprime_after=evals[-1]["dprime"], minutes=(time.time() - t0) / 60, evals=evals)


def check_ab(seeds, n_train, n_eval, eval_every, procs, kw):
    from multiprocessing import Pool
    t0 = time.time()
    jobs = [(s, n_train, n_eval, eval_every, kw) for s in seeds]
    if procs > 1:
        with Pool(min(procs, len(jobs))) as p:
            res = p.map(_ab_worker, jobs)
    else:
        res = [_ab_worker(j) for j in jobs]
    print("\nseed | N | до обучения | после (оценка 1) | после (оценка 2, другой ГСЧ) | d' до -> после | мин")
    for r in res:
        print(f"  {r['seed']} | {r['N']} | {100*r['acc_before']:.1f}% | {100*r['acc_after']:.1f}% | "
              f"{100*r['acc_after_seed2']:.1f}% | {r['dprime_before']:.2f} -> {r['dprime_after']:.2f} | {r['minutes']:.1f}")
    acc = np.array([r["acc_after"] for r in res])
    print(f"среднее после: {100*acc.mean():.1f}% (min {100*acc.min():.1f}%), всего {(time.time()-t0)/60:.1f} мин")
    _dump("ab_results.json", dict(results=res, kw=kw, wall_minutes=(time.time() - t0) / 60))


def _load_trained(seed):
    from oipong.bioconfig import make_bio_config
    from oipong.task_ab import ABTask
    d = json.load(open(os.path.join(OUT, f"ab_seed{seed}_log.json")))
    cfg_d = d["cfg"]
    cfg = make_bio_config(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in cfg_d.items()})
    task = ABTask(cfg)                      # геометрия и маска связей детерминированы seed
    z = np.load(os.path.join(OUT, f"ab_seed{seed}.npz"))
    assert task.net.W.shape == z["W"].shape
    task.net.W[:] = z["W"]
    if "sig_base" in z:
        task.sig_base = z["sig_base"].copy()
    return task, z["W_before"], d


# ----------------------------------------------------------------------------- (в)
def check_viz(seed):
    from oipong import viz
    from oipong.info import sensory_motor_info
    t0 = time.time()
    task, W_before, d = _load_trained(seed)
    mea_trials, labels, choices = viz.record_mea_trials(task, n=120)
    els = np.concatenate(task.el_in)
    elm = np.concatenate(task.el_out)
    info = sensory_motor_info(mea_trials, els, elm, labels, choices)
    print("TE сенсор->мотор: %.4f бит (суррогат %.4f, p=%.3f); TE мотор->сенсор: %.4f (суррогат %.4f, p=%.3f)" % (
        info["TE_sens_to_mot"]["TE"], info["TE_sens_to_mot"]["TE_surrogate"], info["TE_sens_to_mot"]["p_value"],
        info["TE_mot_to_sens"]["TE"], info["TE_mot_to_sens"]["TE_surrogate"], info["TE_mot_to_sens"]["p_value"]))
    print("MI(стимул; ответ): %.4f бит (суррогат %.4f, p=%.3f), точность на этих пробах %.1f%%" % (
        info["MI_stim_choice"]["MI"], info["MI_stim_choice"]["MI_surrogate"], info["MI_stim_choice"]["p_value"],
        100 * np.mean([(c == task.target[l]) for c, l in zip(choices, labels)])))
    rec = viz.record_demo(task, labels=("A", "B"))
    evals = d["evals"]
    viz.dashboard(task, W_before, rec, d["train_log"], evals, info, path=os.path.join(OUT, "dashboard.png"))
    print("сохранено out/dashboard.png")
    saved = viz.animate(task, rec, path_gif=os.path.join(OUT, "dashboard_anim.gif"),
                        path_mp4=os.path.join(OUT, "dashboard_anim.mp4"), max_frames=110)
    print("сохранено:", ", ".join(os.path.relpath(p, HERE) for p in saved))
    _dump("info_metrics.json", info)
    print(f"время: {time.time()-t0:.1f} с")


def check_env(seed, n=60):
    """Обученная сеть при другой температуре, [Mg2+] и вязкости (вязкость меняет сборку, поэтому для неё
    печатается только радиус связности и число синапсов новой сети)."""
    from oipong.bioconfig import make_bio_config
    from oipong.bionet import BioNet
    task, _, _ = _load_trained(seed)
    rows = []
    for T in (30.0, 33.0, 36.0, 39.0):
        task.net.set_temperature(T)
        ev = task.evaluate(n, seed=777)
        r = np.mean([x["rate_RS"] for x in ev["trials"]])
        rows.append(dict(kind="T", value=T, acc=ev["acc"], rate_RS=float(r)))
        print(f"T = {T:4.1f} °C: точность {100*ev['acc']:.1f}%, RS {r:.2f} Гц")
    task.net.set_temperature(36.0)
    for mg in (0.0, 1.0, 2.0):
        task.net.gel.mg = mg
        ev = task.evaluate(n, seed=777)
        r = np.mean([x["rate_RS"] for x in ev["trials"]])
        rows.append(dict(kind="Mg", value=mg, acc=ev["acc"], rate_RS=float(r)))
        print(f"[Mg2+] = {mg:.1f} мМ: точность {100*ev['acc']:.1f}%, RS {r:.2f} Гц")
    task.net.gel.mg = 1.0
    for eta in (0.5, 1.0, 2.0, 4.0):
        net = BioNet(make_bio_config(seed=seed, viscosity=eta, n_neurons=task.cfg.n_neurons))
        s = net.summary()
        rows.append(dict(kind="viscosity", value=eta, conn_radius=s["conn_radius_um"], synapses=s["synapses"]))
        print(f"вязкость x{eta}: радиус связности {s['conn_radius_um']} мкм, синапсов {s['synapses']}")
    _dump("env_checks.json", rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["fi", "ab", "viz", "env"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--train", type=int, default=900)
    ap.add_argument("--eval", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=300)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--kw", type=str, default="{}", help="переопределения BioConfig, JSON")
    a = ap.parse_args()
    if a.what == "fi":
        check_fi()
    elif a.what == "ab":
        kw = json.loads(a.kw)
        kw["n_neurons"] = a.n
        check_ab(a.seeds, a.train, a.eval, a.eval_every, a.procs, kw)
    elif a.what == "viz":
        check_viz(a.seed)
    else:
        check_env(a.seed)
