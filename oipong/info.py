"""Информационные метрики: взаимная информация (MI) и энтропия переноса (TE), в битах.

Оценка «подстановочная» (plug-in) по гистограммам после дискретизации по квантилям. Она смещена вверх
на малых выборках, поэтому рядом всегда печатается та же величина на ПЕРЕМЕШАННЫХ данных (суррогат):
значимо только превышение над суррогатом. Поправка = вычитание среднего по суррогатам.

MI(X;Y) = sum p(x,y) log2 [ p(x,y) / (p(x) p(y)) ]
TE(X->Y) = sum p(y_{t+1}, y_t, x_t) log2 [ p(y_{t+1} | y_t, x_t) / p(y_{t+1} | y_t) ]     (Schreiber 2000,
           Phys. Rev. Lett. 85:461), история длины 1, сдвиг источника lag бинов.
Суррогат для TE: источник берётся из ДРУГОЙ пробы (перестановка проб) — сохраняет автокорреляцию и
стимул-зависимую форму сигнала, рвёт только связь внутри пробы.
"""
import numpy as np


def quantize(x, n_levels=3):
    """Дискретизация по квантилям (равные доли) в 0..n_levels-1."""
    x = np.asarray(x, float)
    if n_levels <= 1 or np.all(x == x.flat[0]):
        return np.zeros(x.shape, dtype=int)
    qs = np.quantile(x, np.linspace(0, 1, n_levels + 1)[1:-1])
    return np.searchsorted(qs, x, side="right").astype(int)


def _entropy_from_counts(c):
    c = c[c > 0].astype(float)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def _joint_entropy(*cols):
    cols = [np.asarray(c, dtype=np.int64) for c in cols]
    key = np.zeros(len(cols[0]), dtype=np.int64)
    for c in cols:
        key = key * (int(c.max()) + 1 if len(c) else 1) + c
    _, cnt = np.unique(key, return_counts=True)
    return _entropy_from_counts(cnt)


def mutual_information(x, y):
    """MI дискретных x, y (биты)."""
    return _joint_entropy(x) + _joint_entropy(y) - _joint_entropy(x, y)


def mi_with_surrogate(x, y, n_shuffle=50, rng=None):
    """MI, среднее MI на перемешанном y и поправленная MI = MI - суррогат."""
    rng = rng or np.random.default_rng(0)
    mi = mutual_information(x, y)
    sur = [mutual_information(x, rng.permutation(y)) for _ in range(n_shuffle)]
    return dict(MI=mi, MI_surrogate=float(np.mean(sur)), MI_corrected=mi - float(np.mean(sur)),
                p_value=float((np.sum(np.array(sur) >= mi) + 1) / (n_shuffle + 1)))


def transfer_entropy(src_trials, dst_trials, lag=1):
    """TE(src -> dst) по набору проб (список 1D дискретных рядов одинаковой длины). Пары берутся только
    внутри пробы (стык проб не используется)."""
    yf, yp, xp = [], [], []
    for x, y in zip(src_trials, dst_trials):
        x, y = np.asarray(x), np.asarray(y)
        T = len(y)
        if T <= lag:
            continue
        yf.append(y[lag:])
        yp.append(y[lag - 1:T - 1])
        xp.append(x[:T - lag])
    yf, yp, xp = np.concatenate(yf), np.concatenate(yp), np.concatenate(xp)
    # TE = H(Yf,Yp) - H(Yp) - H(Yf,Yp,Xp) + H(Yp,Xp)
    return (_joint_entropy(yf, yp) - _joint_entropy(yp) - _joint_entropy(yf, yp, xp) + _joint_entropy(yp, xp))


def te_with_surrogate(src_trials, dst_trials, lag=1, n_shuffle=30, rng=None):
    rng = rng or np.random.default_rng(0)
    te = transfer_entropy(src_trials, dst_trials, lag)
    n = len(src_trials)
    sur = []
    for _ in range(n_shuffle):
        perm = rng.permutation(n)
        sur.append(transfer_entropy([src_trials[i] for i in perm], dst_trials, lag))
    return dict(TE=te, TE_surrogate=float(np.mean(sur)), TE_corrected=te - float(np.mean(sur)),
                p_value=float((np.sum(np.array(sur) >= te) + 1) / (n_shuffle + 1)))


def sensory_motor_info(mea_trials, el_sens, el_mot, labels, choices, n_levels=3, lag=1):
    """Сводка от сенсорных электродов к моторным.
    mea_trials: список массивов (бины x электроды) по пробам; el_sens/el_mot: индексы электродов;
    labels/choices: стимул и ответ в каждой пробе.
    Возвращает: TE(сенсор->мотор), TE(мотор->сенсор) как контроль направления, MI(стимул; ответ),
    MI(стимул; сигнал моторных электродов за пробу)."""
    sens = [m[:, el_sens].sum(axis=1) for m in mea_trials]
    mot = [m[:, el_mot].sum(axis=1) for m in mea_trials]
    allS, allM = np.concatenate(sens), np.concatenate(mot)
    qs = np.quantile(allS, np.linspace(0, 1, n_levels + 1)[1:-1])
    qm = np.quantile(allM, np.linspace(0, 1, n_levels + 1)[1:-1])
    sd = [np.searchsorted(qs, s, side="right") for s in sens]
    md = [np.searchsorted(qm, m, side="right") for m in mot]
    lab = np.array([0 if l == "A" else 1 for l in labels])
    out = dict(TE_sens_to_mot=te_with_surrogate(sd, md, lag), TE_mot_to_sens=te_with_surrogate(md, sd, lag),
               MI_stim_choice=mi_with_surrogate(lab, np.asarray(choices)))
    return out
