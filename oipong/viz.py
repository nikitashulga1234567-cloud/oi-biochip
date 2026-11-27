"""Визуализация (Matplotlib): многопанельная панель V1–V6, сохранение PNG и анимации GIF/MP4.

V1 3D-топология: нейроны X, Y, Z; форма — тип (RS кружок, FS треугольник); цвет — мембранный потенциал;
   спайк — крупная белая обводка. Плоскость HD-MEA (Z = 0) — сетка точек, стимулирующие и моторные кластеры выделены.
V2 растр E/I (RS синие, FS красные) + мгновенная частота сети (Гц, правая ось); серые полосы — стимулы.
V3 тепловая карта HD-MEA (сигнал электродов в бине или за окно стимула).
V4 веса W до/после обучения: средний вес между группами (у A, у B, у out_1, у out_2, остальные).
V5 V(t) и затворы m, h, n, p выбранных нейронов (RS и FS).
V6 баланс E–I (возбуждающий и тормозной токи на RS), кривая обучения и d', параметры среды, TE/MI.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec, animation
from .bionet import Recorder
from .neuron import RS, FS

V_CMAP, V_RANGE = "coolwarm", (-80.0, 0.0)
C_E, C_I = "#2b6cb0", "#c53030"


# ------------------------------------------------------------------ сбор данных
def record_demo(task, labels=("A", "B"), seed=4242, v_every_ms=2.0, gate_ids=None, bin_ms=5.0):
    """Записать несколько проб (без обучения) с полной телеметрией. Состояние сети восстанавливается."""
    net = task.net
    if gate_ids is None:
        r1 = task.read_out[0]
        rs = np.where(net.is_exc)[0]
        fs = np.where(net.is_inh)[0]
        sel = np.abs(net.xyz[:, 0] - task.cfg.stim_A[0]) + np.abs(net.xyz[:, 1] - task.cfg.stim_A[1]) + net.xyz[:, 2]
        gate_ids = [int(rs[np.argmin(sel[rs])]), int(fs[np.argmin(sel[fs])])]
    rec = Recorder(spikes=True, bin_ms=bin_ms, v_every_ms=v_every_ms, gate_ids=gate_ids, k_every_ms=10.0)
    saved = task._save_dynamic()
    rng = np.random.default_rng(seed)
    old = net.rng
    net.rng = rng
    t_start = net.t
    trials = []
    try:
        for lab in labels:
            trials.append(task.trial(lab, learn=False, rng=rng, rec=rec))
    finally:
        net.rng = old
        task._load_dynamic(saved)
    d = rec.arrays()
    d["t0"] = t_start
    d["trials"] = trials
    return d


def record_mea_trials(task, n=100, seed=31337):
    """Пробы без обучения с записью только MEA-бинов: материал для TE/MI. Возвращает (mea_trials, labels, choices)."""
    net = task.net
    saved = task._save_dynamic()
    rng = np.random.default_rng(seed)
    old = net.rng
    net.rng = rng
    labels = np.array(["A", "B"] * (n // 2))
    rng.shuffle(labels)
    out, ch = [], []
    try:
        for lab in labels:
            rec = Recorder(spikes=False, bin_ms=5.0, mea=True, balance=False)
            r = task.trial(str(lab), learn=False, rng=rng, rec=rec)
            out.append(np.array(rec.mea_bins))
            ch.append(r["choice"])
    finally:
        net.rng = old
        task._load_dynamic(saved)
    return out, list(labels), ch


GROUP_NAMES = ["у A", "у B", "у out1", "у out2", "прочие"]


def neuron_groups(task, radius=80.0):
    """Группа каждого RS-нейрона по близости (в плоскости XY) к центрам кластеров; FS не входят (-1)."""
    c, net = task.cfg, task.net
    g = np.full(net.N, 4)
    for k, ctr in enumerate([c.stim_A, c.stim_B, c.out_1, c.out_2]):
        d = np.hypot(net.xyz[:, 0] - ctr[0], net.xyz[:, 1] - ctr[1])
        g[(d <= radius) & (g == 4)] = k
    g[~net.is_exc] = -1
    return g


def group_weight_matrix(task, W, groups=None):
    groups = neuron_groups(task) if groups is None else groups
    M = task.net.mask
    out = np.full((5, 5), np.nan)
    for a in range(5):
        for b in range(5):
            sel = M[np.ix_(groups == a, groups == b)]
            if sel.any():
                out[a, b] = W[np.ix_(groups == a, groups == b)][sel].mean()
    return out


# ------------------------------------------------------------------ панели
def _v1(ax, task, V, spiking=None, title="V1 3D-топология"):
    net, c = task.net, task.cfg
    xyz = net.xyz
    m = task.net.mea
    ax.scatter(m.exy[:, 0], m.exy[:, 1], np.zeros(m.n_el), s=2, c="#999999", alpha=0.25, depthshade=False)
    for el, col in [(task.el_in[0], "#38a169"), (task.el_in[1], "#d69e2e"), (task.el_out[0], "#805ad5"),
                    (task.el_out[1], "#dd6b20")]:
        ax.scatter(m.exy[el, 0], m.exy[el, 1], np.zeros(len(el)), s=14, c=col, depthshade=False, marker="s")
    arts = []
    for t, mk in [(RS, "o"), (FS, "^")]:
        sel = net.types == t
        sizes = np.full(sel.sum(), 7.0)
        edge = np.full(sel.sum(), 0.0)
        if spiking is not None:
            sizes = np.where(spiking[sel], 40.0, 7.0)
            edge = np.where(spiking[sel], 1.2, 0.0)
        sc = ax.scatter(xyz[sel, 0], xyz[sel, 1], xyz[sel, 2], c=V[sel], cmap=V_CMAP, vmin=V_RANGE[0], vmax=V_RANGE[1],
                        s=sizes, marker=mk, edgecolors="white", linewidths=edge, depthshade=False)
        arts.append((sc, sel))
    ax.set_xlim(0, c.box[0]); ax.set_ylim(0, c.box[1]); ax.set_zlim(0, c.box[2])
    ax.set_xlabel("X, мкм", fontsize=7); ax.set_ylabel("Y, мкм", fontsize=7); ax.set_zlabel("Z, мкм", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_title(title, fontsize=9)
    ax.view_init(elev=22, azim=-60)
    ax.text2D(0.0, -0.12, "электроды Z=0: ■A зел. ■B жёлт. ■out1 фиол. ■out2 оранж.; ● RS ▲ FS; белая обводка = спайк",
              transform=ax.transAxes, fontsize=6)
    return arts


def _v2(ax, rec, net, t0=0.0):
    t, i = rec["spk_t"] - t0, rec["spk_i"]
    if len(t):
        e = net.is_exc[i]
        ax.scatter(t[e], i[e], s=0.6, c=C_E, marker="|", linewidths=0.4)
        ax.scatter(t[~e], i[~e], s=0.6, c=C_I, marker="|", linewidths=0.4)
    for a, b, lab in rec["stim_label"]:
        ax.axvspan(a - t0, b - t0, color="0.85", zorder=0)
        ax.text((a + b) / 2 - t0, net.N * 1.01, lab, ha="center", fontsize=7)
    ax.set_xlabel("t, мс", fontsize=7); ax.set_ylabel("нейрон", fontsize=7)
    ax.set_title("V2 растр E (синие) / I (красные) и частота сети", fontsize=9)
    ax.tick_params(labelsize=6)
    ax2 = ax.twinx()
    bt = rec["bins_t"] - t0
    bc = rec["bin_counts"]
    if len(bc):
        rate_e = bc[:, 0] / net.is_exc.sum() / (rec["bin_ms"] / 1000.0)
        rate_i = bc[:, 1] / net.is_inh.sum() / (rec["bin_ms"] / 1000.0)
        ax2.plot(bt, rate_e, color=C_E, lw=1.0, label="RS, Гц")
        ax2.plot(bt, rate_i, color=C_I, lw=1.0, label="FS, Гц")
        ax2.legend(fontsize=6, loc="upper right")
    ax2.set_ylabel("частота, Гц", fontsize=7)
    ax2.tick_params(labelsize=6)
    ax.set_xlim(0, max(bt.max() if len(bt) else 1.0, 1.0))
    return ax2


def _v3(ax, task, img, title="V3 HD-MEA"):
    m = task.net.mea
    p = task.cfg.mea_pitch
    im = ax.imshow(img.T, origin="lower", cmap="magma", extent=[0, m.nx * p, 0, m.ny * p], aspect="equal")
    for ctr, lab in [(task.cfg.stim_A, "A"), (task.cfg.stim_B, "B"), (task.cfg.out_1, "out1"), (task.cfg.out_2, "out2")]:
        ax.add_patch(plt.Circle(ctr, task.cfg.cluster_radius, fill=False, color="cyan", lw=0.8))
        ax.text(ctr[0], ctr[1] + task.cfg.cluster_radius + 8, lab, color="cyan", fontsize=7, ha="center")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("X, мкм", fontsize=7); ax.set_ylabel("Y, мкм", fontsize=7)
    ax.tick_params(labelsize=6)
    return im


def _v4(fig, sub, task, W_before, W_after):
    g = neuron_groups(task)
    mb, ma = group_weight_matrix(task, W_before, g), group_weight_matrix(task, W_after, g)
    vmax = np.nanmax([mb, ma])
    inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=sub, wspace=0.35)
    ims = []
    for k, (M, ttl) in enumerate([(mb, "до"), (ma, "после")]):
        ax = fig.add_subplot(inner[k])
        ims.append(ax.imshow(M * 1000, cmap="viridis", vmin=0, vmax=vmax * 1000))
        ax.set_xticks(range(5)); ax.set_yticks(range(5))
        ax.set_xticklabels(GROUP_NAMES, rotation=60, fontsize=6); ax.set_yticklabels(GROUP_NAMES, fontsize=6)
        ax.set_xlabel("пост", fontsize=7)
        if k == 0:
            ax.set_ylabel("пре", fontsize=7)
        ax.set_title(f"V4 W(E→E) {ttl}, мкСм/см²", fontsize=8)
        for a in range(5):
            for b in range(5):
                if not np.isnan(M[a, b]):
                    ax.text(b, a, f"{M[a,b]*1000:.1f}", ha="center", va="center", fontsize=5.5, color="white")
    return ims


def _v5(fig, sub, rec, t0=0.0, net=None):
    """V(t) и затворы: для каждого выбранного нейрона две оси (V сверху, затворы снизу). У FS g_M = 0,
    поэтому p не рисуется (переменная считается, но тока не даёт)."""
    n = min(2, len(rec["gate_ids"]))
    inner = gridspec.GridSpecFromSubplotSpec(2 * n, 1, subplot_spec=sub, hspace=0.12,
                                             height_ratios=[1.3, 1.0] * n)
    gt = rec["gate_t"] - t0
    axes = []
    for k in range(n):
        gid = rec["gate_ids"][k]
        is_fs = net is not None and net.types[gid] == FS
        nm = "FS" if is_fs else "RS"
        axv = fig.add_subplot(inner[2 * k])
        axv.plot(gt, rec["gates"]["V"][:, k], color="k", lw=0.7)
        axv.set_ylabel(f"{nm} #{gid}\nV, мВ", fontsize=7)
        axv.tick_params(labelsize=6); axv.set_xticklabels([])
        axg = fig.add_subplot(inner[2 * k + 1], sharex=axv)
        for gname, col in [("m", "#e53e3e"), ("h", "#3182ce"), ("n", "#38a169"), ("p", "#d69e2e")]:
            if gname == "p" and is_fs:
                continue
            axg.plot(gt, rec["gates"][gname][:, k], color=col, lw=0.7, label=gname)
        axg.set_ylim(-0.02, 1.02); axg.tick_params(labelsize=6)
        axg.legend(fontsize=5.5, ncol=4, loc="upper right", framealpha=0.6)
        axg.set_ylabel("затворы", fontsize=7)
        if k == 0:
            axv.set_title("V5 V(t) и затворы m, h, n, p (у FS g_M = 0)", fontsize=9)
            axg.set_xticklabels([])
        else:
            axg.set_xlabel("t, мс", fontsize=7)
        axes += [axv, axg]
    return axes


def _v6(fig, sub, task, rec, train_log, evals, info, t0=0.0):
    inner = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=sub, wspace=0.45)
    # баланс E-I
    ax = fig.add_subplot(inner[0])
    if len(rec["bal_t"]):
        ax.plot(rec["bal_t"] - t0, np.array(rec["bal_exc"]) * 1000, color=C_E, lw=0.8, label="g_E (AMPA+NMDA·B)")
        ax.plot(rec["bal_t"] - t0, np.array(rec["bal_inh"]) * 1000, color=C_I, lw=0.8, label="g_I (ГАМК-А)")
        ratio = np.mean(rec["bal_inh"]) / max(np.mean(rec["bal_exc"]), 1e-12)
        ax.set_title(f"V6 баланс E–I на RS: g_I/g_E = {ratio:.2f}", fontsize=8)
        for a, b, lab in rec["stim_label"]:
            ax.axvspan(a - t0, b - t0, color="0.9", zorder=0)
    ax.set_xlabel("t, мс", fontsize=7); ax.set_ylabel("мкСм/см² (среднее по RS)", fontsize=7)
    ax.legend(fontsize=6); ax.tick_params(labelsize=6)
    # обучение и d'
    ax = fig.add_subplot(inner[1])
    if train_log:
        corr = np.array([r["correct"] for r in train_log], float)
        w = max(1, min(50, len(corr)))
        roll = np.convolve(corr, np.ones(w) / w, mode="valid") * 100
        ax.plot(np.arange(w, len(corr) + 1), roll, color="0.5", lw=0.8, label=f"обучение (окно {w})")
    if evals:
        ax.plot([e["trial"] for e in evals], [100 * e["acc"] for e in evals], "o-", color="#2f855a", ms=3,
                label="отложенная оценка")
        axd = ax.twinx()
        axd.plot([e["trial"] for e in evals], [e["dprime"] for e in evals], "s--", color="#805ad5", ms=3)
        axd.set_ylabel("d'", fontsize=7, color="#805ad5"); axd.tick_params(labelsize=6)
    ax.axhline(50, color="r", ls=":", lw=0.7); ax.axhline(70, color="g", ls=":", lw=0.7)
    ax.set_ylim(20, 100)
    ax.set_xlabel("проба", fontsize=7); ax.set_ylabel("точность, %", fontsize=7)
    ax.set_title("точность A/B и d'", fontsize=8); ax.legend(fontsize=5.5, loc="lower right"); ax.tick_params(labelsize=6)
    # среда + TE/MI
    ax = fig.add_subplot(inner[2])
    labels, vals, surs = [], [], []
    if info:
        for key, nm, f in [("TE_sens_to_mot", "TE с→м", "TE"), ("TE_mot_to_sens", "TE м→с", "TE"),
                           ("MI_stim_choice", "MI стим;ответ", "MI")]:
            if key in info:
                labels.append(nm); vals.append(info[key][f]); surs.append(info[key][f + "_surrogate"])
    if labels:
        x = np.arange(len(labels))
        ax.bar(x - 0.18, vals, 0.36, color="#3182ce", label="данные")
        ax.bar(x + 0.18, surs, 0.36, color="0.7", label="суррогат")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6, rotation=20)
        ax.legend(fontsize=6)
    ax.set_ylabel("бит", fontsize=7); ax.tick_params(labelsize=6)
    g = task.net.gel
    c = task.cfg
    ax.set_title("TE/MI и среда", fontsize=8)
    txt = (f"T = {g.T:.1f} °C (φ затворов ×{g.phi_gate:.2f})\n[Mg²⁺] = {g.mg:.1f} мМ\n"
           f"вязкость ×{c.viscosity:.2f} → R связн. {g.conn_radius:.0f} мкм\n"
           f"[K⁺]o ср. {g.K.mean():.2f} / макс {g.K.max():.2f} мМ\nфон {c.bg_rate:.0f} Гц × {c.bg_w*1000:.1f} мкСм/см²")
    ax.text(1.04, 0.98, txt, transform=ax.transAxes, fontsize=6, va="top", family="monospace")


def dashboard(task, W_before, rec, train_log=None, evals=None, info=None, path=None, frame_idx=None):
    """Собрать панель V1–V6 и (если path) сохранить PNG. Возвращает фигуру."""
    net = task.net
    t0 = rec.get("t0", 0.0)
    fig = plt.figure(figsize=(19, 12.5))
    gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[1.15, 1.0, 0.9], hspace=0.42, wspace=0.28)
    # V1: снимок в момент наибольшей активности стимула
    if frame_idx is None and len(rec["v_t"]):
        frame_idx = _peak_frame(rec, net)
    V = rec["v_snap"][frame_idx] if len(rec["v_t"]) else net.nrn.V
    sp = _spiking_near(rec, rec["v_t"][frame_idx], net.N) if len(rec["v_t"]) else None
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    arts = _v1(ax1, task, V, sp, title=f"V1 3D-топология, t = {rec['v_t'][frame_idx]-t0:.0f} мс" if len(rec['v_t']) else "V1")
    cb = fig.colorbar(arts[0][0], ax=ax1, shrink=0.55, pad=0.08)
    cb.set_label("V, мВ", fontsize=7); cb.ax.tick_params(labelsize=6)
    ax2 = fig.add_subplot(gs[0, 1:])
    _v2(ax2, rec, net, t0)
    # V3: средний сигнал за окно стимула A (первая проба)
    ax3 = fig.add_subplot(gs[1, 0])
    img = _mea_window(rec, task, 0)
    im = _v3(ax3, task, img, title="V3 HD-MEA: сигнал за окно стимула " + (rec["stim_label"][0][2] if rec["stim_label"] else ""))
    fig.colorbar(im, ax=ax3, shrink=0.8).ax.tick_params(labelsize=6)
    _v4(fig, gs[1, 1], task, W_before, net.W)
    _v5(fig, gs[1, 2], rec, t0, net)
    _v6(fig, gs[2, :], task, rec, train_log or [], evals or [], info or {}, t0)
    s = net.summary()
    fig.suptitle(f"OI-биочип v4: N = {s['N']} (RS {s['n_RS']}, FS {s['n_FS']}), синапсов {s['synapses']}, "
                 f"электродов {s['electrodes']}, задача A/B", fontsize=11)
    if path:
        fig.savefig(path, dpi=110, bbox_inches="tight")
    return fig


def _peak_frame(rec, net):
    if not len(rec["stim_label"]):
        return len(rec["v_t"]) // 2
    a, b, _ = rec["stim_label"][0]
    vt = rec["v_t"]
    st = rec["spk_t"]
    best, best_n = 0, -1
    for k, t in enumerate(vt):
        if a <= t <= b:
            n = np.sum((st > t - 2.0) & (st <= t))
            if n > best_n:
                best, best_n = k, n
    return best


def _spiking_near(rec, t, N, win=2.0):
    st, si = rec["spk_t"], rec["spk_i"]
    sp = np.zeros(N, dtype=bool)
    sel = (st > t - win) & (st <= t)
    sp[si[sel]] = True
    return sp


def _mea_window(rec, task, k):
    if not len(rec["mea_bins"]):
        return np.zeros((task.net.mea.nx, task.net.mea.ny))
    a, b, _ = rec["stim_label"][k] if rec["stim_label"] else (rec["bins_t"][0], rec["bins_t"][-1], "")
    sel = (rec["bins_t"] > a) & (rec["bins_t"] <= b + task.cfg.decision_lag)
    return task.net.mea.to_image(rec["mea_bins"][sel].sum(axis=0))


# ------------------------------------------------------------------ анимация
def animate(task, rec, path_gif=None, path_mp4=None, max_frames=120, fps=12):
    """Анимация: V1 (потенциалы и спайки в 3D), V2 (бегущий курсор), V3 (MEA в текущем бине), V5 (курсор).
    Сохраняет GIF (Pillow) и/или MP4 (ffmpeg, если есть). Возвращает список сохранённых путей."""
    net = task.net
    t0 = rec.get("t0", 0.0)
    vt = rec["v_t"]
    idx = np.linspace(0, len(vt) - 1, min(max_frames, len(vt))).astype(int)
    fig = plt.figure(figsize=(18, 9))
    gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1.15, 1.0, 0.8], hspace=0.38, wspace=0.3)
    ax1 = fig.add_subplot(gs[:, 0], projection="3d")
    arts = _v1(ax1, task, rec["v_snap"][idx[0]], _spiking_near(rec, vt[idx[0]], net.N))
    cb = fig.colorbar(arts[0][0], ax=ax1, shrink=0.45, pad=0.06)
    cb.set_label("V, мВ", fontsize=7); cb.ax.tick_params(labelsize=6)
    ax2 = fig.add_subplot(gs[0, 1])
    _v2(ax2, rec, net, t0)
    cursors = [ax2.axvline(0, color="k", lw=1.0)]
    ax3 = fig.add_subplot(gs[0, 2])
    mb, bt = rec["mea_bins"], rec["bins_t"]
    vmax = np.percentile(mb, 99.5) if len(mb) else 1.0
    im = _v3(ax3, task, task.net.mea.to_image(mb[0]) if len(mb) else np.zeros((2, 2)), title="V3 HD-MEA, бин 5 мс")
    im.set_clim(0, max(vmax, 1e-6))
    for a in _v5(fig, gs[1, 1], rec, t0, net):
        cursors.append(a.axvline(0, color="k", lw=0.8))
    ax6 = fig.add_subplot(gs[1, 2])
    if len(rec["bal_t"]):
        ax6.plot(rec["bal_t"] - t0, np.array(rec["bal_exc"]) * 1000, color=C_E, lw=0.8, label="g_E")
        ax6.plot(rec["bal_t"] - t0, np.array(rec["bal_inh"]) * 1000, color=C_I, lw=0.8, label="g_I")
        ax6.legend(fontsize=6)
    ax6.set_title("V6 баланс E–I на RS, мкСм/см²", fontsize=9); ax6.tick_params(labelsize=6)
    ax6.set_xlabel("t, мс", fontsize=7)
    cursors.append(ax6.axvline(0, color="k", lw=0.8))
    title = fig.suptitle("", fontsize=11)

    def update(k):
        f = idx[k]
        t = vt[f]
        sp = _spiking_near(rec, t, net.N)
        for sc, sel in arts:
            sc.set_array(rec["v_snap"][f][sel])
            sc.set_sizes(np.where(sp[sel], 40.0, 7.0))
            sc.set_linewidths(np.where(sp[sel], 1.2, 0.0))
        for cu in cursors:
            cu.set_xdata([t - t0, t - t0])
        if len(bt):
            j = int(np.clip(np.searchsorted(bt, t), 0, len(bt) - 1))
            im.set_data(task.net.mea.to_image(mb[j]).T)
        lab = ""
        for a, b, l in rec["stim_label"]:
            if a <= t <= b:
                lab = f" | стимул {l}"
        title.set_text(f"t = {t - t0:6.1f} мс{lab}")
        return []

    anim = animation.FuncAnimation(fig, update, frames=len(idx), interval=1000 / fps, blit=False)
    saved = []
    if path_gif:
        anim.save(path_gif, writer=animation.PillowWriter(fps=fps), dpi=70)
        saved.append(path_gif)
    if path_mp4 and animation.writers.is_available("ffmpeg"):
        anim.save(path_mp4, writer=animation.FFMpegWriter(fps=fps, bitrate=1800), dpi=90)
        saved.append(path_mp4)
    plt.close(fig)
    return saved


def fi_figure(fi_rs, fi_fs, traces=None, path=None):
    """f–I кривые RS и FS (средняя, первая и последняя мгновенная частота) + примеры трасс."""
    fig, axes = plt.subplots(1, 3 if traces else 1, figsize=(15 if traces else 5, 4))
    axes = np.atleast_1d(axes)
    ax = axes[0]
    for rows, col, nm in [(fi_rs, C_E, "RS"), (fi_fs, C_I, "FS")]:
        I = [r["I_uA_cm2"] for r in rows]
        ax.plot(I, [r["rate_Hz"] for r in rows], "o-", color=col, label=f"{nm} средняя")
        ax.plot(I, [r["f_first_Hz"] for r in rows], "--", color=col, alpha=0.6, label=f"{nm} 1/ISI первый")
        ax.plot(I, [r["f_last_Hz"] for r in rows], ":", color=col, alpha=0.9, label=f"{nm} 1/ISI последний")
    ax.set_xlabel("ток, мкА/см²"); ax.set_ylabel("частота, Гц"); ax.legend(fontsize=7); ax.set_title("f–I (1 с ступенька)")
    if traces:
        for k, (nm, tr) in enumerate(traces.items()):
            inner = axes[k + 1].get_subplotspec().subgridspec(2, 1, hspace=0.08, height_ratios=[1.3, 1.0])
            axes[k + 1].remove()
            av = fig.add_subplot(inner[0])
            ag = fig.add_subplot(inner[1], sharex=av)
            av.plot(tr["t"], tr["V"], color="k", lw=0.7)
            av.set_title(nm, fontsize=9); av.set_ylabel("V, мВ"); av.tick_params(labelbottom=False)
            for g, col in [("m", "#e53e3e"), ("h", "#3182ce"), ("n", "#38a169"), ("p", "#d69e2e")]:
                if g == "p" and nm.startswith("FS"):
                    continue
                ag.plot(tr["t"], tr[g], color=col, lw=0.7, label=g)
            ag.set_ylim(-0.02, 1.02); ag.set_xlabel("t, мс"); ag.set_ylabel("затворы")
            ag.legend(fontsize=7, ncol=4, loc="upper right")
            av.set_xlim(0, 350)
    if path:
        fig.savefig(path, dpi=110, bbox_inches="tight")
    return fig
