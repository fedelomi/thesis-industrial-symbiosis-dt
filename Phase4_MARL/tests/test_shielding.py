"""Unit tests for the Yazdanpanah-style shielding layer."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from config.scenarios import DC_PROFILES, MFG_PROFILES
from env.shielding import ShieldingLayer, UPGRADE_INDEX_INVERSE


@pytest.fixture
def shielding() -> ShieldingLayer:
    return ShieldingLayer()


@pytest.fixture
def dc_mid():
    return DC_PROFILES["Mid_LC"]


@pytest.fixture
def mfg_mid():
    # MidT_04_PaperPulp_Medium -> requires HP, T_req 90 C
    return MFG_PROFILES["MidT"]


def test_clip_q_over_available(shielding, dc_mid, mfg_mid) -> None:
    action = np.array([dc_mid.q_available_kw * 1.5, 90.0, 100.0, 5, 1], dtype=float)
    clipped, violations = shielding.apply(action, dc_mid, mfg_mid)
    assert clipped[0] == pytest.approx(dc_mid.q_available_kw)
    assert any("BLOCKED" in v and "Q_negotiated" in v for v in violations)


def test_clip_t_below_pinch(shielding, dc_mid, mfg_mid) -> None:
    action = np.array([1000.0, mfg_mid.t_req_c - 10.0, 100.0, 5, 1], dtype=float)
    clipped, violations = shielding.apply(action, dc_mid, mfg_mid)
    assert clipped[1] == pytest.approx(mfg_mid.t_req_c - 5.0)
    assert any("BLOCKED" in v and "T_supply_offered" in v for v in violations)


def test_warn_price_below_floor(shielding, dc_mid, mfg_mid) -> None:
    action = np.array([1000.0, 90.0, 0.0, 5, 1], dtype=float)
    clipped, violations = shielding.apply(action, dc_mid, mfg_mid)
    assert clipped[2] > 0.0
    assert any("WARNING" in v and "amortised floor" in v for v in violations)


def test_block_upgrade_mismatch(shielding, dc_mid, mfg_mid) -> None:
    # mfg_mid requires HP (idx=1); we offer direct_HX (idx=0)
    action = np.array([1000.0, 90.0, 100.0, 5, 0], dtype=float)
    clipped, violations = shielding.apply(action, dc_mid, mfg_mid)
    assert int(round(clipped[4])) == UPGRADE_INDEX_INVERSE[mfg_mid.upgrade_required]
    assert any("BLOCKED" in v and "upgrade_tech" in v for v in violations)


def test_clean_action_passes(shielding, dc_mid, mfg_mid) -> None:
    action = np.array([1000.0, 95.0, 200.0, 10, 1], dtype=float)
    clipped, violations = shielding.apply(action, dc_mid, mfg_mid)
    assert not any(v.startswith("BLOCKED") for v in violations)
    np.testing.assert_allclose(clipped[:4], action[:4])


def test_logger_emits_warning(shielding, dc_mid, mfg_mid, caplog) -> None:
    caplog.set_level(logging.WARNING, logger="phase4.shielding")
    action = np.array([dc_mid.q_available_kw * 2, 95.0, 200.0, 5, 1], dtype=float)
    shielding.apply(action, dc_mid, mfg_mid)
    assert any("BLOCKED" in rec.message for rec in caplog.records)
