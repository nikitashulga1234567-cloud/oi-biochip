"""Структура сети: 3D-геометрия, роли нейронов, связность, нейрогенез/апоптоз, прунинг/спраутинг.
Роли хранятся в массиве role (а не в списках индексов), поэтому вход/выход не «съезжают» при удалении."""
import numpy as np
import pandas as pd
from .config import ROLE_SENS, ROLE_HID, ROLE_MOT, ROLE_NAME
from .hh import REST

PER_NEURON = ["xyz", "v", "m", "h", "n", "ge", "gi", "role", "nid", "inhib", "rate", "above", "pavg", "born",
              "mgroup", "pref"]
PER_PAIR = ["W", "E", "mask", "C"]


class Network:
    def __init__(self, cfg, rng):
        self.cfg = cfg
        self.rng = rng
        self.box = np.array(cfg.box)
        self.next_id = 0
        self.episode = 0
        self.events = []
        self._build()

    # ------------------------------------------------------------ базовое
    @property
    def N(self):
        return len(self.v)

    def idx(self, role):
        return np.where(self.role == role)[0]

    @property
    def mot_up(self):
        return np.where((self.role == ROLE_MOT) & (self.mgroup == 0))[0]

    @property
    def mot_down(self):
        return np.where((self.role == ROLE_MOT) & (self.mgroup == 1))[0]

    def _alloc(self, n):
        self.xyz = np.zeros((n, 3))
        self.v = np.full(n, REST["v"])
        self.m = np.full(n, REST["m"])
        self.h = np.full(n, REST["h"])
        self.n = np.full(n, REST["n"])
        self.ge, self.gi = np.zeros(n), np.zeros(n)
        self.role = np.zeros(n, dtype=int)
        self.nid = np.zeros(n, dtype=int)
        self.inhib = np.zeros(n, dtype=bool)
        self.rate = np.zeros(n)
        self.above = np.zeros(n, dtype=bool)
        self.pavg = np.full(n, 0.3)
        self.born = np.zeros(n, dtype=int)
        self.mgroup = np.full(n, -1)
        self.pref = np.full(n, np.nan)
        self.W = np.zeros((n, n))
        self.E = np.zeros((n, n))
        self.C = np.zeros((n, n))
        self.mask = np.zeros((n, n), dtype=bool)

    def _build(self):
        c, rng, box = self.cfg, self.rng, self.box
        ns, nh, nm = c.n_sens, c.n_hid0, 2 * c.n_mot_each
        n = ns + nh + nm
        self._alloc(n)
        placed = []

        def place(lo_x, hi_x):
            p = None
            for _ in range(500):
                p = np.array([rng.uniform(lo_x, hi_x), rng.uniform(20, box[1] - 20), rng.uniform(20, box[2] - 20)])
                if not placed or np.min(np.linalg.norm(np.array(placed) - p, axis=1)) >= 40.0:
                    break
            placed.append(p)
            return p

        prefs = np.linspace(-0.85, 0.85, ns)
        for i in range(n):
            if i < ns:
                self.role[i] = ROLE_SENS
                self.xyz[i] = place(15, 70)
                self.pref[i] = prefs[i]
            elif i < ns + nh:
                self.role[i] = ROLE_HID
                self.xyz[i] = place(90, box[0] - 90)
            else:
                self.role[i] = ROLE_MOT
                self.xyz[i] = place(box[0] - 70, box[0] - 15)
                self.mgroup[i] = 0 if i < ns + nh + c.n_mot_each else 1
            self.nid[i] = self.next_id
            self.next_id += 1
        self.inhib = (self.role == ROLE_HID) & (rng.random(n) < c.inhib_frac)
        self._connect_initial()

    def allowed(self):
        a = np.outer(self.role != ROLE_MOT, self.role != ROLE_SENS)
        np.fill_diagonal(a, False)
        return a

    def dist(self):
        return np.linalg.norm(self.xyz[:, None, :] - self.xyz[None, :, :], axis=2)

    def _connect_initial(self):
        c, rng = self.cfg, self.rng
        d, allowed = self.dist(), self.allowed()
        p = np.clip(np.exp(-d / c.conn_lambda) * (d <= c.conn_radius) * allowed * 1.6, 0, 0.9)
        mask = (rng.random(p.shape) < p) & allowed
        if self.idx(ROLE_HID).size:
            for i in self.idx(ROLE_SENS):
                cand = np.where(allowed[i] & (self.role == ROLE_HID))[0]
                mask[i, cand[np.argsort(d[i, cand])][:c.min_deg]] = True
            for j in self.idx(ROLE_MOT):
                cand = np.where(allowed[:, j] & (self.role == ROLE_HID))[0]
                mask[cand[np.argsort(d[cand, j])][:c.min_deg], j] = True
        # длинные проекции входного электродного массива прямо в моторную зону
        mask &= ~np.outer(self.role == ROLE_SENS, self.role == ROLE_MOT)
        sm = np.outer(self.role == ROLE_SENS, self.role == ROLE_MOT)
        mask |= sm & (rng.random(mask.shape) < c.direct_p)
        if c.direct_p <= 0:      # чисто многослойная сеть: никаких коротких путей сенсор -> мотор
            mask &= ~sm
        self.mask = mask
        self.W = np.where(mask, rng.uniform(c.w_init[0], c.w_init[1], mask.shape), 0.0)
        self.scale_inputs()

    def scale_inputs(self):
        """Синаптическое масштабирование: сумма входящих возбуждающих весов <= w_in_cap."""
        exc = ~self.inhib
        tot = (self.W * exc[:, None]).sum(axis=0)
        over = tot > self.cfg.w_in_cap
        if over.any():
            f = np.where(over, self.cfg.w_in_cap / np.maximum(tot, 1e-9), 1.0)
            self.W[exc, :] *= f[None, :]

    # ------------------------------------------------- структурная пластичность
    def _append(self, xyz, role, inhib, mgroup=-1, pref=np.nan):
        n = self.N
        fills = dict(v=REST["v"], m=REST["m"], h=REST["h"], n=REST["n"], ge=0.0, gi=0.0, role=role,
                     nid=self.next_id, inhib=inhib, rate=0.0, above=False, pavg=0.3, born=self.episode,
                     mgroup=mgroup, pref=pref)
        self.xyz = np.vstack([self.xyz, xyz])
        for k, val in fills.items():
            setattr(self, k, np.append(getattr(self, k), val))
        for k in PER_PAIR:
            a = getattr(self, k)
            new = np.zeros((n + 1, n + 1), dtype=a.dtype)
            new[:n, :n] = a
            setattr(self, k, new)
        self.next_id += 1
        return n

    def _remove(self, i):
        for k in PER_NEURON:
            setattr(self, k, np.delete(getattr(self, k), i, axis=0))
        for k in PER_PAIR:
            setattr(self, k, np.delete(np.delete(getattr(self, k), i, axis=0), i, axis=1))

    def log_event(self, kind, nid, note):
        self.events.append(dict(Episode=self.episode, Event=kind, Neuron_ID=nid, Note=note,
                                Hidden=int((self.role == ROLE_HID).sum()), Total=self.N))

    def neurogenesis(self):
        hid = self.idx(ROLE_HID)
        if len(hid) == 0 or len(hid) >= self.cfg.hid_max:
            return None
        util = self.rate[hid] * (self.W[hid, :].sum(axis=1) + 0.05)
        parent = hid[int(np.argmax(util + 1e-6 * self.rng.random(len(hid))))]
        xyz = np.clip(self.xyz[parent] + self.rng.normal(0, 18.0, 3), 10, self.box - 10)
        child = self._append(xyz, ROLE_HID, bool(self.inhib[parent]))
        self.rate[child] = 0.5 * self.rate[parent]
        for j in range(child):
            if self.mask[parent, j]:
                self.mask[child, j] = True
                self.W[child, j] = self.W[parent, j] * self.rng.uniform(0.7, 1.2)
            if self.mask[j, parent]:
                self.mask[j, child] = True
                self.W[j, child] = self.W[j, parent] * self.rng.uniform(0.7, 1.2)
        self.mask &= self.allowed()
        self.W *= self.mask
        self.scale_inputs()
        self.log_event("neurogenesis", int(self.nid[child]), f"родитель {int(self.nid[parent])}")
        return child

    def apoptosis(self):
        hid = self.idx(ROLE_HID)
        if len(hid) <= self.cfg.hid_min:
            return None
        hid = hid[(self.episode - self.born[hid]) >= self.cfg.grace_episodes]
        if len(hid) == 0:
            return None
        util = self.rate[hid] * (self.W[hid, :].sum(axis=1) + 0.05) + 0.02 * self.W[:, hid].sum(axis=0)
        victim = hid[int(np.argmin(util))]
        nid = int(self.nid[victim])
        self._remove(victim)
        self.log_event("apoptosis", nid, "низкая полезность")
        return victim

    def structural_update(self, rolling_win):
        c = self.cfg
        weak = self.mask & (self.W < c.prune_w)
        if weak.any():
            n_weak = int(weak.sum())
            self.mask[weak] = False
            self.W[weak] = 0.0
            self.log_event("synapse_prune", -1, f"{n_weak} синапсов")
        allowed = self.allowed() & ~self.mask & (self.dist() <= c.conn_radius)
        cand = np.where(allowed, self.C, -1.0)
        for _ in range(3):
            k = np.unravel_index(np.argmax(cand), cand.shape)
            if cand[k] <= 1e-4:
                break
            self.mask[k] = True
            self.W[k] = 0.08
            cand[k] = -1.0
            self.log_event("synapse_sprout", int(self.nid[k[0]]), f"-> {int(self.nid[k[1]])}")
        if rolling_win < c.grow_below:
            self.neurogenesis()
        hid = self.idx(ROLE_HID)
        if len(hid) > c.hid_min and (self.rate[hid] < 0.01).any():
            self.apoptosis()
        self.rate *= 0.5
        self.scale_inputs()

    # ---------------------------------------------------------- таблицы
    def class_w(self, pre_role, post_role):
        m = self.mask & np.outer(self.role == pre_role, self.role == post_role)
        return float(self.W[m].mean()) if m.any() else 0.0

    def mean_w(self):
        return float(self.W[self.mask].mean()) if self.mask.any() else 0.0

    def neurons_table(self):
        return pd.DataFrame(dict(Neuron_ID=self.nid, Role=[ROLE_NAME[r] for r in self.role],
                                 Inhibitory=self.inhib, X_um=self.xyz[:, 0].round(1), Y_um=self.xyz[:, 1].round(1),
                                 Z_um=self.xyz[:, 2].round(1), Activity=self.rate.round(3),
                                 Out_Weight=self.W.sum(axis=1).round(3), In_Weight=self.W.sum(axis=0).round(3)))

    def synapses_table(self):
        i, j = np.where(self.mask)
        return pd.DataFrame(dict(Pre_ID=self.nid[i], Post_ID=self.nid[j],
                                 Pre_Role=[ROLE_NAME[r] for r in self.role[i]],
                                 Post_Role=[ROLE_NAME[r] for r in self.role[j]],
                                 Weight=self.W[i, j].round(4), Inhibitory=self.inhib[i]))

    def state(self):
        d = {k: getattr(self, k).copy() for k in PER_NEURON + PER_PAIR}
        d.update(next_id=self.next_id, events=list(self.events))
        return d

    def load(self, d):
        for k in PER_NEURON + PER_PAIR:
            if k in d:
                setattr(self, k, d[k].copy())
        self.next_id = int(d["next_id"])
        self.events = list(d["events"])
