import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PyQt6 import QtWidgets

from antenna_response_tab import (
    AntennaResponseTab,
    AntennaRunResult,
    MicrowaveSettings,
    analyze_run_folder,
    educontrol_difference_warning,
    find_rabi_runs,
    fit_classification_reasons,
    fit_failure_reasons,
    read_microwave_settings,
)


def make_synthetic_run(parent, name="2026-11-20-12-00-00-rabi"):
    run = parent / name
    eval_dir = run / "seq0" / "eval"
    src_dir = run / "seq0" / "src"
    eval_dir.mkdir(parents=True)
    src_dir.mkdir(parents=True)

    time_ns = np.linspace(0.0, 1000.0, 201)
    time_s = time_ns * 1e-9
    frequency_hz = 8e6
    signal = 0.98 + 0.04 * np.exp(-time_s / 2e-6) * np.cos(
        2 * np.pi * frequency_hz * time_s + 0.2
    )
    signal += np.random.default_rng(42).normal(0.0, 1e-4, signal.size)

    dtype = np.dtype([
        ("pulseDuration", "f8"),
        ("sig", "f8"),
        ("ref", "f8"),
        ("contrast", "f8"),
    ])
    data = np.zeros((1, 1, time_ns.size), dtype=dtype)
    data["pulseDuration"] = time_ns
    data["sig"] = signal
    data["ref"] = 1.0
    data["contrast"] = signal
    with h5py.File(eval_dir / "rabi.h5", "w") as handle:
        handle.create_dataset("data", data=data)

    scalars = np.zeros((1, 1), dtype=[("rabi:freq0", "f8"), ("rabi:t_pi", "f8")])
    scalars["rabi:freq0"] = 8.1e6
    scalars["rabi:t_pi"] = 1 / (2 * 8.1e6)
    with h5py.File(run / "scalarData.h5", "w") as handle:
        handle.create_dataset("data", data=scalars)

    (src_dir / "rabi.seq").write_text(
        "- iterate 1:\n"
        "  - inline: |\n"
        "      $freq = 2875000000.0\n"
        "      $mwAmp = 0.2\n"
        "      $df = 2000000.0\n",
        encoding="utf-8",
    )
    return run


def test_settings_use_frequency_plus_detuning(tmp_path):
    run = make_synthetic_run(tmp_path)
    settings = read_microwave_settings(run)

    assert settings.frequency_hz == 2.875e9
    assert settings.detuning_hz == 2e6
    assert settings.driven_frequency_hz == 2.877e9
    assert settings.amplitude == 0.2


def test_synthetic_rabi_run_is_loaded_and_fitted(tmp_path):
    run = make_synthetic_run(tmp_path)
    result = analyze_run_folder(run)

    assert result.error is None
    assert result.settings.driven_frequency_hz == 2.877e9
    assert result.educontrol_rabi_hz == 8.1e6
    assert result.rabi_hz == pytest.approx(8e6, rel=0.01)
    assert result.rabi_err_hz / result.rabi_hz < 0.01
    assert result.pi_s == pytest.approx(62.5e-9, rel=0.01)
    assert result.r_squared > 0.99
    assert result.periods_in_window == pytest.approx(8.0, rel=0.01)
    assert result.failure_reasons == []
    assert result.exclusion_reasons == []
    assert result.warning_reasons == []
    assert result.classification == "accepted"


def test_run_discovery_and_classification_thresholds(tmp_path):
    first = make_synthetic_run(tmp_path / "day1")
    second = make_synthetic_run(tmp_path / "day2", "2026-11-20-13-00-00-rabi")

    assert find_rabi_runs(tmp_path) == [first, second]
    assert "uncertainty is zero or nonfinite" in fit_failure_reasons(8e6, 0.0)
    assert fit_failure_reasons(0.5e6, 0.01e6) == []

    excluded, warnings = fit_classification_reasons(8e6, 0.81e6, 3.0)
    assert "relative uncertainty above 10%" in excluded
    assert warnings == []
    excluded, warnings = fit_classification_reasons(8e6, 0.01e6, 0.9)
    assert "fewer than 1 fitted period in the window" in excluded
    excluded, warnings = fit_classification_reasons(8e6, 0.48e6, 1.5)
    assert excluded == []
    assert "relative uncertainty from 5% to 10%" in warnings
    assert "only 1 to 2 fitted periods in the window" in warnings

    assert educontrol_difference_warning(8e6, 10e6) == "fit differs from EduControl by 20.0%"
    assert educontrol_difference_warning(8e6, 8.5e6) is None


def test_period_classification_boundaries():
    frequency = 10e6
    uncertainty = 0.01e6

    excluded, warnings = fit_classification_reasons(frequency, uncertainty, 0.999)
    assert "fewer than 1 fitted period in the window" in excluded
    excluded, warnings = fit_classification_reasons(frequency, uncertainty, 1.0)
    assert excluded == []
    assert "only 1 to 2 fitted periods in the window" in warnings
    excluded, warnings = fit_classification_reasons(frequency, uncertainty, 1.001)
    assert excluded == []
    assert "only 1 to 2 fitted periods in the window" in warnings

    excluded, warnings = fit_classification_reasons(frequency, uncertainty, 1.999)
    assert "only 1 to 2 fitted periods in the window" in warnings
    excluded, warnings = fit_classification_reasons(frequency, uncertainty, 2.0)
    assert excluded == []
    assert warnings == []
    excluded, warnings = fit_classification_reasons(frequency, uncertainty, 2.001)
    assert excluded == []
    assert warnings == []


def test_relative_uncertainty_classification_boundaries():
    frequency = 10e6

    excluded, warnings = fit_classification_reasons(frequency, 0.0499 * frequency, 3.0)
    assert excluded == []
    assert warnings == []
    excluded, warnings = fit_classification_reasons(frequency, 0.05 * frequency, 3.0)
    assert excluded == []
    assert warnings == []
    excluded, warnings = fit_classification_reasons(frequency, 0.0501 * frequency, 3.0)
    assert excluded == []
    assert "relative uncertainty from 5% to 10%" in warnings

    excluded, warnings = fit_classification_reasons(frequency, 0.0999 * frequency, 3.0)
    assert excluded == []
    assert "relative uncertainty from 5% to 10%" in warnings
    excluded, warnings = fit_classification_reasons(frequency, 0.10 * frequency, 3.0)
    assert excluded == []
    assert "relative uncertainty from 5% to 10%" in warnings
    excluded, warnings = fit_classification_reasons(frequency, 0.1001 * frequency, 3.0)
    assert "relative uncertainty above 10%" in excluded


def test_educontrol_difference_warning_boundary():
    educontrol = 10e6

    assert educontrol_difference_warning(9.001e6, educontrol) is None  # 9.99%
    assert educontrol_difference_warning(9.0e6, educontrol) is None  # exactly 10%
    warning = educontrol_difference_warning(8.999e6, educontrol)  # 10.01%
    assert warning == "fit differs from EduControl by 10.0%"


def test_tab_opens_and_populates_synthetic_run(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    run = make_synthetic_run(tmp_path)
    tab = AntennaResponseTab()
    tab.add_runs([run])

    assert app is not None
    assert tab.table.rowCount() == 1
    assert tab.table.item(0, 2).text() == "2.877000"
    assert tab.table.item(0, 5).text() == "8.1000"
    assert tab.table.item(0, 0).checkState().value == 2
    assert tab.table.item(0, 7).text().startswith("ACCEPTED:")
    tab.close()


def test_excluded_runs_stay_checked_and_amplitude_groups_use_date_and_frequency():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    tab = AntennaResponseTab()

    def result(name, frequency_hz, amplitude, excluded=False):
        item = AntennaRunResult(
            path=Path(name),
            settings=MicrowaveSettings(
                frequency_hz=frequency_hz,
                detuning_hz=0.0,
                amplitude=amplitude,
            ),
            rabi_hz=5e6,
            rabi_err_hz=0.05e6,
            pi_s=100e-9,
            r_squared=0.9,
            periods_in_window=4.0,
        )
        if excluded:
            item.exclusion_reasons = ["fewer than 1 fitted period in the window"]
        return item

    tab.results = [
        result("2026-11-20-10-00-00-rabi", 2.815e9, 0.1),
        result("2026-11-20-10-01-00-rabi", 2.815e9, 0.2),
        result("2026-11-21-10-00-00-rabi", 2.815e9, 0.3),
        result("2026-11-20-10-02-00-rabi", 2.815e9, 0.01, excluded=True),
    ]
    tab._populate_table()

    assert app is not None
    assert tab.table.item(3, 0).checkState().value == 2
    assert tab.table.item(3, 7).text().startswith("EXCLUDED:")
    # Two accepted groups: one for each acquisition date. The excluded series is gray and separate.
    assert len(tab.amplitude_plot.listDataItems()) == 3
    tab.close()
