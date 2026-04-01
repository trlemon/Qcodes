import warnings

import numpy as np
import pytest

from qcodes.instrument_drivers.tektronix import TektronixAWG5014
from qcodes.utils.deprecate import QCoDeSDeprecationWarning


@pytest.fixture(scope="function")
def awg():
    awg_sim = TektronixAWG5014(
        "awg_sim",
        address="GPIB0::1::INSTR",
        timeout=1,
        terminator="\n",
        pyvisa_sim_file="Tektronix_AWG5014C.yaml",
    )
    yield awg_sim

    awg_sim.close()


def test_init_awg(awg) -> None:
    idn_dict = awg.IDN()

    assert idn_dict["vendor"] == "QCoDeS"


def test_pack_waveform(awg) -> None:
    N = 25

    rng = np.random.default_rng()
    waveform = rng.random(N)
    m1 = rng.integers(0, 2, N)
    m2 = rng.integers(0, 2, N)

    package = awg._pack_waveform(waveform, m1, m2)

    assert package is not None


def test_make_awg_file(awg) -> None:
    N = 25

    rng = np.random.default_rng()
    waveforms = [[rng.random(N)]]
    m1s = [[rng.integers(0, 2, N)]]
    m2s = [[rng.integers(0, 2, N)]]
    nreps = [1]
    trig_waits = [0]
    goto_states = [0]
    jump_tos = [0]

    awgfile = awg.make_awg_file(
        waveforms,
        m1s,
        m2s,
        nreps,
        trig_waits,
        goto_states,
        jump_tos,
        preservechannelsettings=False,
    )

    assert len(awgfile) > 0


class TestLegacyChannelAttributes:
    """Tests that the old flat ch{i}_* attribute names still work
    but emit a QCoDeSDeprecationWarning."""

    CHANNEL_PARAMS = (
        "state",
        "amp",
        "offset",
        "waveform",
        "direct_output",
        "add_input",
        "filter",
        "DC_out",
    )
    MARKER_PARAMS = (("del", "delay"), ("high", "high"), ("low", "low"))

    def test_legacy_channel_param_exists(self, awg) -> None:
        """All old ch{i}_{param} names resolve to the correct parameter."""
        for i in range(1, 5):
            for param in self.CHANNEL_PARAMS:
                old_name = f"ch{i}_{param}"
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", QCoDeSDeprecationWarning)
                    old_attr = getattr(awg, old_name)
                new_attr = getattr(awg.submodules[f"ch{i}"], param)
                assert old_attr is new_attr, (
                    f"{old_name} did not resolve to ch{i}.{param}"
                )

    def test_legacy_marker_param_exists(self, awg) -> None:
        """All old ch{i}_m{j}_{param} names resolve to the correct parameter."""
        for i in range(1, 5):
            for j in (1, 2):
                for old_suffix, new_name in self.MARKER_PARAMS:
                    old_name = f"ch{i}_m{j}_{old_suffix}"
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", QCoDeSDeprecationWarning)
                        old_attr = getattr(awg, old_name)
                    new_attr = getattr(
                        awg.submodules[f"ch{i}"].submodules[f"m{j}"], new_name
                    )
                    assert old_attr is new_attr, (
                        f"{old_name} did not resolve to ch{i}.m{j}.{new_name}"
                    )

    def test_legacy_channel_param_warns(self, awg) -> None:
        """Accessing an old channel param name emits QCoDeSDeprecationWarning."""
        with pytest.warns(QCoDeSDeprecationWarning, match="ch1_amp.*ch1.amp"):
            _ = awg.ch1_amp

    def test_legacy_marker_param_warns(self, awg) -> None:
        """Accessing an old marker param name emits QCoDeSDeprecationWarning."""
        with pytest.warns(QCoDeSDeprecationWarning, match="ch2_m1_high.*ch2.m1.high"):
            _ = awg.ch2_m1_high

    def test_legacy_marker_del_warns(self, awg) -> None:
        """The renamed 'del' -> 'delay' param emits a correct warning."""
        with pytest.warns(QCoDeSDeprecationWarning, match="ch3_m2_del.*ch3.m2.delay"):
            _ = awg.ch3_m2_del

    def test_nonexistent_attr_raises(self, awg) -> None:
        """An attribute that doesn't match any legacy name still raises."""
        with pytest.raises(AttributeError, match="no_such_attr"):
            _ = awg.no_such_attr

    def test_nonexistent_legacy_style_raises(self, awg) -> None:
        """A ch{i}_* name that doesn't map to a real param still raises."""
        with pytest.raises(AttributeError):
            _ = awg.ch1_bogus_param
