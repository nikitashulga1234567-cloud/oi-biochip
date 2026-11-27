"""Конфигурация, пресеты, сохранение/загрузка настроек и журнал запусков."""
import json
import time

ROLE_SENS, ROLE_HID, ROLE_MOT = 0, 1, 2
ROLE_NAME = {0: "sensory", 1: "hidden", 2: "motor"}


class Config:
    """Все параметры в одном месте. Меняйте через make_config(preset, **kw)."""
    seed = 7

    # ---- геометрия и состав сети
    box = (800.0, 800.0, 600.0)     # мкм, гидрогелевая матрица
    n_sens = 8                       # сенсорные электроды
    n_hid0 = 14                      # скрытые нейроны на старте
    n_mot_each = 8                   # нейронов в группе UP и в группе DOWN
    hid_min, hid_max = 8, 36
    direct_p = 1.0                   # доля прямых связей сенсор -> мотор
    min_deg = 4
    w_init = (0.15, 0.5)
    conn_radius = 230.0              # мкм
    conn_lambda = 110.0
    inhib_frac = 0.2

    # ---- время
    frame_ms = 10.0                  # один игровой кадр
    dt = 0.05                        # шаг интегрирования HH, мс

    # ---- игра
    ball_vx = 1.0 / 12.0
    paddle_gain = 0.015              # смещение ракетки за 1 спайк разницы UP-DOWN
    paddle_max = 0.09
    hit_half = 0.12
    encoding = "signed"              # 'signed' (порог) | 'gauss' (колокол)
    enc_thr = 0.25
    enc_span = 0.15
    tuning_sigma = 0.35
    i_sens_max = 22.0                # мкА/см2

    # ---- шум и смещения
    hid_bias = 0.0
    noise_sigma = 0.5                # фоновый шум (скрытые)
    mot_bias = 0.0                   # моторные при ОЦЕНКЕ
    mot_noise = 0.5
    explore_noise = 2.0              # моторные при ОБУЧЕНИИ (исследование); None = как mot_noise
    explore_bias = 0.0
    explore_hid_noise = None         # шум скрытых при ОБУЧЕНИИ (нужен для обучения скрытого слоя); None = как noise_sigma
    feedback_noise = False           # шум-штраф на сенсорах после промаха (идея DishBrain)

    # ---- синапсы
    g_scale = 0.3                    # мСм/см2 на единицу веса W
    w_max = 0.8
    w_in_cap = 1.6
    syn_tau_e, syn_tau_i = 5.0, 10.0

    # ---- обучение (RM-Hebb: dW = lr * da * pre * (post - <post>))
    rule = "rmhebb"                  # 'rmhebb' (рекомендуется) | 'legacy' (как в v2.1, для сравнения)
    lr = 0.15
    lr_hid_mult = 1.0                # множитель lr для синапсов НА скрытые нейроны
    e_decay = 0.0                    # память следа между кадрами (0 = только текущий кадр)
    pre_cur_w = 1.0                  # вес пресинаптических спайков ТЕКУЩЕГО кадра
    pre_prev_w = 0.0                 # вес спайков ПРОШЛОГО кадра
    shaping = 4.0                    # награда за сокращение ошибки движением ракетки
    in_window_bonus = 0.0
    r_hit, r_miss = 0.0, 0.0
    use_baseline = False             # вычитать скользящую среднюю награды (в v2.1 было True: давало дрейф весов)
    da_eps = 1e-6                    # |da| ниже порога: обновления нет (защита от дрейфа в паузах)
    pavg_tau = 0.02

    # ---- структурная пластичность
    structural = False
    struct_every = 10
    grow_below = 0.95
    prune_w = 0.01
    grace_episodes = 60              # защита новорождённых от апоптоза

    # ---- химико-термическая среда (BioEnvironment)
    env_enabled = True
    temperature = 37.0               # °C
    temp_ref = 37.0                  # температура, при которой калиброваны параметры HH
    q10_gate = 3.0                   # кинетика воротных переменных m, h, n
    q10_cond = 1.4                   # максимальные проводимости ионных каналов
    q10_syn = 2.0                    # кинетика синапса (скорость спада)
    # пулы нейромедиаторов, относительные единицы (1.0 = референсный фон)
    glu_tonic, gaba_tonic, da_tonic = 1.0, 1.0, 1.0
    glu_release, gaba_release = 0.002, 0.002     # прирост за один спайк
    da_release = 0.15                # прирост фазического дофамина на единицу |награды|
    glu_tau, gaba_tau, da_tau = 30.0, 50.0, 80.0  # мс, клиренс (обратный захват)
    env_k = 1.0                      # полунасыщение функции Хилла
    da_gate_plasticity = True        # дофамин управляет скоростью обучения


PRESETS = {
    "direct": dict(n_hid0=0, direct_p=1.0, structural=False),
    "hidden": dict(n_hid0=14, direct_p=0.5, structural=False),
    "hidden_structural": dict(n_hid0=14, direct_p=0.5, structural=True),
    "two_layer": dict(n_hid0=24, direct_p=0.0, structural=False, explore_hid_noise=2.0),
    "legacy_v21": dict(n_hid0=0, direct_p=1.0, rule="legacy", e_decay=0.8, pre_prev_w=1.0, use_baseline=True,
                       in_window_bonus=0.1, r_hit=0.6, r_miss=-0.6, env_enabled=False, feedback_noise=True),
}


def make_config(preset="direct", **kw):
    """Config с пресетом и переопределениями. Опечатка в имени параметра -> ошибка."""
    c = Config()
    for k, v in {**PRESETS[preset], **kw}.items():
        if not hasattr(c, k):
            raise KeyError(f"Неизвестный параметр Config: {k}")
        setattr(c, k, v)
    return c


def cfg_to_dict(cfg):
    d = {}
    for k in dir(Config):
        if k.startswith("_") or callable(getattr(Config, k)):
            continue
        v = getattr(cfg, k)
        d[k] = list(v) if isinstance(v, tuple) else v
    return d


def cfg_from_dict(d):
    c = Config()
    for k, v in d.items():
        if hasattr(c, k):
            setattr(c, k, tuple(v) if isinstance(getattr(Config, k), tuple) else v)
    return c


def save_config(cfg, path="experiment_cfg.json"):
    json.dump(cfg_to_dict(cfg), open(path, "w"), ensure_ascii=False, indent=1)


def load_config(path="experiment_cfg.json"):
    return cfg_from_dict(json.load(open(path)))


def log_run(record, path="oi_runs.jsonl"):
    rec = dict(record)
    rec["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
