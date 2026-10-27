"""Overdamped Brownian (low-Reynolds) physics in PymunkProcess.

Checks that a passive cell diffuses with the translational and rotational
diffusion coefficients predicted from temperature, viscosity and the cell's
geometry (Einstein / fluctuation-dissipation), and that it actually rotates —
rotational thermal motion the inertial model never produced.
"""
import math

from viva_munk import core_import
from viva_munk.processes.multibody import PymunkProcess, build_microbe, make_rng


def _trajectory(brownian=True, seed=3, ticks=400, dt=1.0, env=400.0):
    core = core_import()
    p = PymunkProcess({'env_size': env, 'substeps': 20, 'brownian': brownian,
                       'temperature': 300.0, 'viscosity': 1e-3}, core=core)
    rng = make_rng(seed)
    aid, cell = build_microbe(rng, env, agent_id='c0', x=env / 2, y=env / 2,
                              angle=0.3, length=2.0, radius=0.5, density=0.02,
                              velocity=(0, 0), speed_range=(0, 0))
    x = y = env / 2
    ang = 0.3
    xs, ys, angs = [x], [y], [ang]
    for _ in range(ticks):
        out = p.update({'segment_cells': {aid: {**cell, 'location': (x, y), 'angle': ang}}}, dt)
        d = out['segment_cells'][aid]
        dx, dy = d.get('location', (0.0, 0.0))
        x += dx
        y += dy
        ang += d.get('angle', 0.0)
        xs.append(x)
        ys.append(y)
        angs.append(ang)
    return p, xs, ys, angs, dt


def _msd(a, lag):
    return sum((a[i + lag] - a[i]) ** 2 for i in range(len(a) - lag)) / (len(a) - lag)


def test_translational_and_rotational_diffusion_match_theory():
    p, xs, ys, angs, dt = _trajectory()
    d_t_exp, d_r_exp = p._brownian_diffusion(0.5, 2.0)
    # E. coli-scale coefficients, not the ad-hoc inertial jitter.
    assert 0.1 < d_t_exp < 1.0
    assert 0.02 < d_r_exp < 1.0

    lag = 10
    t = lag * dt
    d_t_meas = (_msd(xs, lag) + _msd(ys, lag)) / (4 * t)     # 2-D: MSD = 4 D_t t
    d_r_meas = _msd(angs, lag) / (2 * t)                     # <Δθ²> = 2 D_r t
    # Finite-sample: within ~2x of theory (short lag, single trajectory).
    assert 0.5 * d_t_exp < d_t_meas < 2.0 * d_t_exp, (d_t_meas, d_t_exp)
    assert 0.5 * d_r_exp < d_r_meas < 2.0 * d_r_exp, (d_r_meas, d_r_exp)


def test_cell_rotates_under_thermal_noise():
    _, _, _, angs, _ = _trajectory(ticks=200)
    # Rotational thermal motion exists at all (the inertial model gave none).
    assert abs(angs[-1] - angs[0]) > 0.1


def test_larger_cells_diffuse_slower():
    core = core_import()
    p = PymunkProcess({'env_size': 100, 'brownian': True}, core=core)
    d_t_small, d_r_small = p._brownian_diffusion(0.5, 2.0)
    d_t_big, d_r_big = p._brownian_diffusion(0.5, 6.0)   # longer rod
    assert d_t_big < d_t_small      # more drag -> slower translation
    assert d_r_big < d_r_small      # rotational drag ~ L^3


def test_default_is_inertial_backcompat():
    assert PymunkProcess.config_schema['brownian']['_default'] is False


def test_drag_scale_calibrates_diffusion():
    """The calibration knobs scale diffusion as 1/scale, independently for
    translation and rotation (rotational_drag_scale ~4 brings a 2µm×0.5µm cell's
    D_r into the passive E. coli band ~0.05 rad²/s)."""
    core = core_import()
    base = PymunkProcess({'env_size': 50, 'brownian': True}, core=core)
    d_t0, d_r0 = base._brownian_diffusion(0.5, 2.0)

    rot = PymunkProcess({'env_size': 50, 'brownian': True,
                         'rotational_drag_scale': 4.0}, core=core)
    d_t_r, d_r_r = rot._brownian_diffusion(0.5, 2.0)
    assert abs(d_t_r - d_t0) < 1e-9                 # translation unchanged
    assert abs(d_r_r - d_r0 / 4.0) < 1e-6           # rotation scaled 1/4
    assert 0.03 <= d_r_r <= 0.06                    # in the passive band

    trans = PymunkProcess({'env_size': 50, 'brownian': True,
                           'translational_drag_scale': 2.0}, core=core)
    d_t_t, d_r_t = trans._brownian_diffusion(0.5, 2.0)
    assert abs(d_t_t - d_t0 / 2.0) < 1e-6           # translation scaled 1/2
    assert abs(d_r_t - d_r0) < 1e-9                 # rotation unchanged
