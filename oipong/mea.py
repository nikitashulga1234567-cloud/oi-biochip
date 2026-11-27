"""HD-MEA: матрица электродов в плоскости Z = 0 под культурой.

Считывание. Электрод e видит спайк нейрона i с весом, затухающим с расстоянием r_ei (в 3D, поэтому
глубина Z нейрона входит автоматически):
        a_ei = 1 / (1 + (r_ei / r0)^2)
Это ДОПУЩЕНИЕ: точечный источник тока в однородной среде даёт 1/r (Nunez & Srinivasan 2006), а реальный
спайк — диполь, его амплитуда падает круче (~1/r^2 и быстрее); спайки на практике видны в пределах
~100 мкм от электрода (Buzsáki et al. 2012, Nat. Rev. Neurosci. 13:407). Форма 1/(1+(r/r0)^2) с r0 = 35 мкм
даёт 1 у самого электрода, 0.5 на 35 мкм и ~0.11 на 100 мкм. Сигнал электрода в бин = сумма a_ei * спайки_i + шум.

Стимуляция. Электрод с амплитудой I0 создаёт у нейрона i деполяризующий ток (ТЗ, гауссово поле):
        I_i = I0 * exp(-r_ei^2 / (2 sigma^2))         мкА/см2
Реальная внеклеточная стимуляция действует через «активирующую функцию» (вторую производную потенциала
вдоль аксона, Rattay 1989); гауссова деполяризация — её грубая замена (допущение ТЗ).
"""
import numpy as np


class HDMEA:
    def __init__(self, cfg, xyz):
        self.cfg = cfg
        bx, by = cfg.box[0], cfg.box[1]
        p = cfg.mea_pitch
        xs = np.arange(p / 2, bx, p)
        ys = np.arange(p / 2, by, p)
        self.nx, self.ny = len(xs), len(ys)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        self.exy = np.stack([gx.ravel(), gy.ravel()], axis=1)           # (E, 2), Z = 0
        self.n_el = len(self.exy)
        self.set_neurons(xyz)

    def set_neurons(self, xyz):
        self.xyz = xyz
        d2 = ((self.exy[:, None, 0] - xyz[None, :, 0]) ** 2 + (self.exy[:, None, 1] - xyz[None, :, 1]) ** 2
              + xyz[None, :, 2] ** 2)
        self.r2 = d2                                                     # (E, N), мкм2
        self.A = 1.0 / (1.0 + d2 / self.cfg.mea_r0 ** 2)                 # матрица считывания

    def electrodes_near(self, center, radius):
        """Индексы электродов в радиусе radius от точки center (x, y)."""
        d = np.hypot(self.exy[:, 0] - center[0], self.exy[:, 1] - center[1])
        idx = np.where(d <= radius)[0]
        return idx if len(idx) else np.array([int(np.argmin(d))])

    def read(self, spikes, rng=None):
        """Сигнал всех электродов для вектора спайков (или числа спайков за бин) нейронов."""
        s = self.A @ spikes
        if rng is not None and self.cfg.mea_noise > 0:
            s = s + rng.normal(0.0, self.cfg.mea_noise, s.shape)
        return s

    def stim_current(self, centers_xy, amp, sigma=None):
        """Ток деполяризации (мкА/см2) каждого нейрона от набора точек стимуляции (x, y) на Z = 0."""
        sigma = self.cfg.stim_sigma if sigma is None else sigma
        I = np.zeros(len(self.xyz))
        for cx, cy in np.atleast_2d(centers_xy):
            r2 = (self.xyz[:, 0] - cx) ** 2 + (self.xyz[:, 1] - cy) ** 2 + self.xyz[:, 2] ** 2
            I += amp * np.exp(-r2 / (2.0 * sigma ** 2))
        return I

    def stim_from_electrodes(self, el_idx, amp, sigma=None):
        return self.stim_current(self.exy[el_idx], amp, sigma)

    def to_image(self, sig):
        """Вектор сигналов электродов -> 2D-массив (nx, ny) для тепловой карты."""
        return np.asarray(sig).reshape(self.nx, self.ny)
