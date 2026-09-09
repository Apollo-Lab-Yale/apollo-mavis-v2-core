"""DAgger core-type tests (design doc 01-core §9, §18)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from apollo_mavis_v2_core.dagger.types import (
    CheckpointInfo,
    ControlMode,
    EpisodeSummary,
    FrameAnnotations,
    GateEvent,
    TrainerStatus,
)


def test_control_mode_to_int8_mapping():
    assert ControlMode.POLICY.to_int8() == 0
    assert ControlMode.HUMAN.to_int8() == 1
    assert ControlMode.TAKEOVER_TRANSITION.to_int8() == 2
    # str Enum: wire values round-trip
    assert ControlMode("takeover_transition") is ControlMode.TAKEOVER_TRANSITION
    assert ControlMode.POLICY.value == "policy"
    assert {m.to_int8() for m in ControlMode} == {0, 1, 2}


def _checkpoint() -> CheckpointInfo:
    return CheckpointInfo(
        run_id="run0",
        version=3,
        path="checkpoints/run0/v000003/",
        parent_version=2,
        trained_on_frames=1200,
        trained_on_episodes=[0, 1, 2],
        action_frame="arm_base:arm0",
        action_space="delta_ee",
        sanity_ok=True,
        mean_loss=0.021,
        sha256="a" * 64,
        created_wallclock_ns=1_756_000_000_000_000_000,
    )


def test_dataclasses_are_frozen():
    ev = GateEvent(arm_id="arm0", mode=ControlMode.HUMAN, t_mono=1.5, seq=7, source="keyboard")
    ann = FrameAnnotations(
        control_mode=ControlMode.POLICY,
        executed_action=np.zeros(8, dtype=np.float32),
        policy_action=None,
        policy_version=3,
        action_frame="world",
    )
    summary = EpisodeSummary(
        episode_index=4,
        n_frames=500,
        n_intervention_frames=60,
        n_label_frames=50,
        takeover_segments=2,
        segment_doubts=[0.1, 0.4],
        success=True,
    )
    for obj, field, value in [
        (ev, "seq", 8),
        (ann, "policy_version", 4),
        (_checkpoint(), "version", 9),
        (summary, "n_frames", 0),
    ]:
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(obj, field, value)


def test_frame_annotations_carries_counterfactual():
    ann = FrameAnnotations(
        control_mode=ControlMode.HUMAN,
        executed_action=np.ones(8, dtype=np.float32),
        policy_action=np.zeros(8, dtype=np.float32),
        policy_version=1,
        action_frame="arm_base:arm0",
    )
    assert ann.policy_action is not None
    assert ann.control_mode.to_int8() == 1


def test_trainer_status_round_trip():
    st = TrainerStatus(
        state="training",
        steps_total=1200,
        last_burst_loss=0.034,
        last_checkpoint_version=4,
        last_checkpoint_ts=812.5,
        new_label_frames=37,
    )
    assert TrainerStatus.model_validate_json(st.model_dump_json()) == st

    defaults = TrainerStatus(state="idle")
    assert defaults.steps_total == 0
    assert defaults.last_burst_loss is None
    assert defaults.last_checkpoint_version is None
    assert defaults.new_label_frames == 0
    assert TrainerStatus.model_validate(defaults.model_dump()) == defaults

    with pytest.raises(ValueError):
        TrainerStatus(state="exploded")


def test_episode_summary_actor_counts_are_additive():
    """15-online-dagger §4: ``n_expert_frames`` / ``n_novice_frames`` (the ``actor`` split)
    default to 0 so every pre-phase-14 constructor call still works; appended last."""
    fields = [f.name for f in dataclasses.fields(EpisodeSummary)]
    assert fields[-3:] == ["episode_id", "n_expert_frames", "n_novice_frames"]
    summary = EpisodeSummary(
        episode_index=4, n_frames=500, n_intervention_frames=60, n_label_frames=50,
        takeover_segments=2, segment_doubts=[0.1, 0.4], success=None,
    )
    assert (summary.n_expert_frames, summary.n_novice_frames) == (0, 0)
    split = dataclasses.replace(summary, n_expert_frames=60, n_novice_frames=440)
    assert split.n_expert_frames + split.n_novice_frames == split.n_frames
    assert split.n_expert_frames >= split.n_label_frames  # expert counts transition frames too
    with pytest.raises(dataclasses.FrozenInstanceError):
        split.n_expert_frames = 0
