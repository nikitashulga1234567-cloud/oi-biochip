"""Физика Pong и кодирование наблюдения. Ничего не знает о нейронах."""
import numpy as np


class PongGame:
    def __init__(self, cfg):
        self.cfg = cfg
        self.ball_x = self.ball_y = self.ball_vy = 0.0
        self.paddle_y = 0.5

    def reset(self, rng):
        self.ball_x = 0.0
        self.ball_y = float(rng.uniform(0.1, 0.9))
        self.ball_vy = float(rng.choice([-1, 1]) * rng.uniform(0.0, 0.04))
        self.paddle_y = float(rng.uniform(0.2, 0.8))

    @property
    def error(self):
        return self.ball_y - self.paddle_y

    def encode(self, pref):
        """Активность каждого сенсорного нейрона [0..1] по ошибке мяч - ракетка."""
        c, err = self.cfg, self.error
        if c.encoding == "signed":
            p = np.nan_to_num(pref)
            return np.clip((np.sign(p) * err - c.enc_thr * np.abs(p)) / c.enc_span, 0.0, 1.0)
        return np.exp(-0.5 * ((err - pref) / c.tuning_sigma) ** 2)

    def advance(self, move):
        """Двигает ракетку и мяч на один кадр. Возвращает (paddle_old, done, hit)."""
        c = self.cfg
        paddle_old = self.paddle_y
        self.paddle_y = float(np.clip(self.paddle_y + move, 0.0, 1.0))
        self.ball_x += c.ball_vx
        self.ball_y += self.ball_vy
        if self.ball_y < 0.0 or self.ball_y > 1.0:
            self.ball_vy *= -1.0
            self.ball_y = float(np.clip(self.ball_y, 0.0, 1.0))
        done = self.ball_x >= 1.0
        hit = bool(done and abs(self.error) <= c.hit_half)
        return paddle_old, done, hit

    def shaping_reward(self, paddle_old):
        """Эффект ИМЕННО движения ракетки: 0, если ракетка не двигалась (естественный baseline)."""
        c = self.cfg
        return c.shaping * (abs(self.ball_y - paddle_old) - abs(self.error)) + \
            c.in_window_bonus * float(abs(self.error) <= c.hit_half)
