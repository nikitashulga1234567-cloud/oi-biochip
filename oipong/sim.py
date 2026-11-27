"""OIPong: оркестратор. Собирает Network + PongGame + BioEnvironment + правило обучения."""
import pickle
import numpy as np
import pandas as pd
from .config import Config, ROLE_SENS, ROLE_HID, ROLE_MOT, cfg_to_dict, cfg_from_dict
from .network import Network
from .game import PongGame
from .environment import BioEnvironment
from .hh import integrate_frame
from .learning import apply_plasticity


class OIPong:
    def __init__(self, cfg=None):
        self.cfg = cfg or Config()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.net = Network(self.cfg, self.rng)
        self.game = PongGame(self.cfg)
        self.env = BioEnvironment(self.cfg)
        self.episode = 0
        self.global_frame = 0
        self.reward_baseline = 0.0
        self.noise_left = 0
        self.prev_spk = np.zeros(self.net.N)
        self.game.reset(self.rng)

    # все свойства сети (W, mask, xyz, pref, idx, mot_up, N, events ...) доступны прямо на OIPong
    def __getattr__(self, name):
        if name in ("net", "__setstate__"):
            raise AttributeError(name)
        return getattr(self.net, name)

    # ------------------------------------------------------------------ кадр
    def _sensory_current(self):
        c, net = self.cfg, self.net
        I = np.zeros(net.N)
        s = net.idx(ROLE_SENS)
        I[s] = c.i_sens_max * self.game.encode(net.pref)[s]
        if self.noise_left > 0 and c.feedback_noise:
            I[s] += self.rng.normal(0, c.i_sens_max * 0.5, len(s))
            self.noise_left -= 1
        return I

    def step(self, learn=True):
        c, net, env = self.cfg, self.net, self.env
        spikes = integrate_frame(net, c, self._sensory_current(), self.rng, env, explore=learn)
        up, down = spikes[net.mot_up].sum(), spikes[net.mot_down].sum()
        move = float(np.clip(c.paddle_gain * (up - down), -c.paddle_max, c.paddle_max))
        action = "UP" if move > 0 else ("DOWN" if move < 0 else "HOLD")

        paddle_old, done, hit = self.game.advance(move)
        reward = self.game.shaping_reward(paddle_old)
        if done:
            reward += c.r_hit if hit else c.r_miss

        da = reward - (self.reward_baseline if c.use_baseline else 0.0)
        if learn:
            if c.use_baseline:
                self.reward_baseline += 0.02 * (reward - self.reward_baseline)
            apply_plasticity(net, c, spikes, self.prev_spk, da, lr_gain=env.plasticity_gain)
            net.C = 0.98 * net.C + 0.02 * np.outer(self.prev_spk > 0, spikes > 0)
            net.rate = 0.95 * net.rate + 0.05 * (spikes > 0)
        exc = ~net.inhib
        env.step(c.frame_ms, n_exc_spikes=float(spikes[exc].sum()), n_inh_spikes=float(spikes[net.inhib].sum()),
                 reward=reward)
        self.prev_spk = spikes.copy()
        self.global_frame += 1
        g = self.game
        return dict(spikes=spikes, up=float(up), down=float(down), action=action, move=move, reward=float(reward),
                    da=float(da), done=done, hit=hit, ball_x=g.ball_x, ball_y=g.ball_y, paddle_y=g.paddle_y,
                    error=float(g.error), v=net.v.copy())

    def play_episode(self, learn=True, record=False):
        self.game.reset(self.rng)
        self.prev_spk = np.zeros(self.net.N)
        frames, log, t = [], [], 0
        while True:
            r = self.step(learn=learn)
            snap = self.env.snapshot()
            log.append(dict(Episode=self.episode, Frame=t, Ball_X=round(r["ball_x"], 4), Ball_Y=round(r["ball_y"], 4),
                            Paddle_Y=round(r["paddle_y"], 4), Error=round(r["error"], 4), Spikes_UP=r["up"],
                            Spikes_DOWN=r["down"], Action=r["action"], Reward=round(r["reward"], 3),
                            Dopamine_Signal=round(r["da"], 3), Total_Spikes=float(r["spikes"].sum()),
                            Hit=r["hit"] if r["done"] else None, Temp_C=snap["T"], Glu=snap["Glu"],
                            GABA=snap["GABA"], DA_Level=snap["DA"], Plasticity_Gain=snap["plasticity_gain"]))
            if record:
                frames.append(dict(ball_x=r["ball_x"], ball_y=r["ball_y"], paddle_y=r["paddle_y"],
                                   spikes=r["spikes"].copy(), v=r["v"], action=r["action"]))
            t += 1
            if r["done"]:
                break
        if (not r["hit"]) and learn:
            self.noise_left = 3
        return r["hit"], r["error"], frames, log

    def evaluate(self, n=100, seed=777):
        """n эпизодов БЕЗ обучения и БЕЗ шума исследования; фиксированный seed; состояние обучения не меняется."""
        saved = (self.rng, self.noise_left, self.env.state(), self.reward_baseline)
        self.rng = np.random.default_rng(seed)
        self.noise_left = 0
        hits, errs = [], []
        try:
            for _ in range(n):
                hit, err, _, _ = self.play_episode(learn=False)
                hits.append(bool(hit))
                errs.append(abs(err))
        finally:
            self.rng, self.noise_left, env_state, self.reward_baseline = saved
            self.env.load(env_state)
        return dict(win=float(np.mean(hits)), miss_dist=float(np.mean(errs)), n=int(n))

    # --------------------------------------------------------------- обучение
    def train(self, target=1.0, max_episodes=4000, eval_every=100, eval_n=100, confirm_n=300, window=30,
              log_every=100, frame_log_every=20, eval_seed=777, verbose=True):
        """Обучаем, пока ОЦЕНКА (без обучения и без шума) не достигнет target; затем подтверждение на другом
        seed. max_episodes — эпизодов ЭТОГО запуска (None = без лимита). target=None: без ранней остановки.
        Возвращает (episodes_df, frames_df, eval_df, reached)."""
        hist, frame_log, evals, hits = [], [], [], []
        reached, start = False, self.episode

        def eval_row(extra=None):
            r = self.evaluate(eval_n, seed=eval_seed)
            row = dict(Episode=self.episode, Eval_Win_Pct=round(100 * r["win"], 2), Eval_N=r["n"],
                       Eval_Miss_Distance=round(r["miss_dist"], 4), Neurons_Total=self.net.N,
                       Synapses=int(self.net.mask.sum()))
            if extra:
                row.update(extra)
            return row

        evals.append(eval_row())
        if verbose:
            print(f"эпизод {self.episode:5d} | eval = {evals[-1]['Eval_Win_Pct']:5.1f}% (до обучения в этом запуске)")
        while True:
            self.episode += 1
            self.net.episode = self.episode
            hit, err, _, flog = self.play_episode(learn=True)
            if frame_log_every and self.episode % frame_log_every == 0:
                frame_log.extend(flog)
            hits.append(1 if hit else 0)
            roll, cum = float(np.mean(hits[-window:])), float(np.mean(hits))
            hist.append(self._episode_row(hit, err, roll, cum, float(sum(f["Reward"] for f in flog))))
            if self.cfg.structural and self.episode % self.cfg.struct_every == 0 and len(hits) >= window:
                self.net.structural_update(roll)
            if eval_every and self.episode % eval_every == 0:
                row = eval_row()
                evals.append(row)
                if verbose and self.episode % log_every == 0:
                    print(f"эпизод {self.episode:5d} | eval = {row['Eval_Win_Pct']:5.1f}% | train-окно({window}) = "
                          f"{roll*100:5.1f}% | нейронов {self.net.N} | синапсов {int(self.net.mask.sum())} | "
                          f"ср.вес {self.net.mean_w():.3f}")
                if target is not None and row["Eval_Win_Pct"] >= 100 * target - 1e-9:
                    conf = self.evaluate(confirm_n, seed=eval_seed + 1)
                    row["Confirm_Win_Pct"], row["Confirm_N"] = round(100 * conf["win"], 2), conf["n"]
                    if verbose:
                        print(f"  подтверждение на независимом seed: {100*conf['win']:.1f}% ({confirm_n} эп.)")
                    if conf["win"] >= target - 1e-9:
                        reached = True
                        break
            if max_episodes is not None and self.episode - start >= max_episodes:
                break
        if verbose:
            tag = "ЦЕЛЬ ДОСТИГНУТА" if reached else "остановлено по лимиту эпизодов"
            print(f"\n{tag}: всего эпизодов {self.episode}, последняя оценка {evals[-1]['Eval_Win_Pct']:.1f}%")
        return pd.DataFrame(hist), pd.DataFrame(frame_log), pd.DataFrame(evals), reached

    def _episode_row(self, hit, err, roll, cum, ep_reward=0.0):
        net, hid = self.net, self.net.idx(ROLE_HID)
        snap = self.env.snapshot()
        return dict(Episode=self.episode, Hit=int(hit), Miss_Distance=round(abs(err), 4),
                    Episode_Reward=round(ep_reward, 3), WinRate_Rolling_Pct=round(roll * 100, 2),
                    WinRate_Cumulative_Pct=round(cum * 100, 2), Neurons_Total=net.N, Neurons_Hidden=len(hid),
                    Synapses=int(net.mask.sum()), Mean_Weight=round(net.mean_w(), 4),
                    W_Sens_Hid=round(net.class_w(ROLE_SENS, ROLE_HID), 4),
                    W_Hid_Hid=round(net.class_w(ROLE_HID, ROLE_HID), 4),
                    W_Hid_Mot=round(net.class_w(ROLE_HID, ROLE_MOT), 4),
                    W_Sens_Mot=round(net.class_w(ROLE_SENS, ROLE_MOT), 4),
                    Rate_Sens=round(float(net.rate[net.role == ROLE_SENS].mean()), 3),
                    Rate_Hid=round(float(net.rate[hid].mean()) if len(hid) else 0.0, 3),
                    Rate_Mot=round(float(net.rate[net.role == ROLE_MOT].mean()), 3),
                    Reward_Baseline=round(self.reward_baseline, 4), Temp_C=snap["T"], Glu=snap["Glu"],
                    GABA=snap["GABA"], DA_Level=snap["DA"], Plasticity_Gain=snap["plasticity_gain"])

    # --------------------------------------------------------- сохранение
    def state_dict(self):
        return dict(net=self.net.state(), env=self.env.state(), episode=self.episode,
                    reward_baseline=self.reward_baseline)

    def load_state(self, d):
        self.net.load(d["net"])
        self.env.load(d["env"])
        self.episode = int(d["episode"])
        self.net.episode = self.episode
        self.reward_baseline = float(d["reward_baseline"])
        self.prev_spk = np.zeros(self.net.N)
        self.game.reset(self.rng)


def save_game(game, path="oi_state.pkl"):
    with open(path, "wb") as f:
        pickle.dump(dict(cfg=cfg_to_dict(game.cfg), state=game.state_dict()), f)


def load_game(path="oi_state.pkl"):
    with open(path, "rb") as f:
        d = pickle.load(f)
    game = OIPong(cfg_from_dict(d["cfg"]))
    game.load_state(d["state"])
    return game
