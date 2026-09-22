from __future__ import annotations

import pytest

from minimax_refpack import nodes, rewrite_pins


def test_combo_starts_with_default_then_named_pins():
    values = rewrite_pins.combo_values()
    assert values[0] == rewrite_pins.DEFAULT
    assert tuple(values[1:]) == rewrite_pins.NAMES


def test_every_named_pin_file_is_nonempty():
    for name in rewrite_pins.NAMES:
        text = rewrite_pins.load(name)
        assert text.strip(), name


def test_unknown_pin_lists_known_names():
    with pytest.raises(ValueError, match="rub"):
        rewrite_pins.load("nope")


def test_named_pin_loads_file_and_forces_standard():
    text, job_type = rewrite_pins.apply("", "rub")
    assert "wild kissing" in text
    assert job_type == "standard"


def test_system_prompt_widget_wins_over_named_pin():
    text, job_type = rewrite_pins.apply("custom ⚙", "rub")
    assert text == "custom ⚙"
    assert job_type is None


def test_default_pin_leaves_packaged_path():
    text, job_type = rewrite_pins.apply("", "default")
    assert text == ""
    assert job_type is None
    text, job_type = rewrite_pins.apply("", "graph")
    assert text == ""
    assert job_type is None


def test_rewrite_pin_is_appended_last_on_the_node():
    spec = nodes.MiniMaxH3ReferencePack.INPUT_TYPES()
    ordered = list(spec["required"]) + list(spec["optional"])
    assert ordered[-1] == "rewrite_pin"
    assert ordered[-2] == "original_prompt"
    combo = spec["optional"]["rewrite_pin"][0]
    assert combo == rewrite_pins.combo_values()
    assert combo[0] == "default"


def test_t2v_describe_pin_bans_reference_tags_and_asks_for_three_fields():
    """The pin's whole reason to exist, asserted on the file itself.

    It cannot prove the VLM obeys - only that nobody edited the instruction out.
    The three-field order is T2VA's (docs/i2v-best-practices.md in the MiniMax
    pack); a tag in the output would point at an asset a text-only render never
    receives.
    """
    text = rewrite_pins.load("t2v_describe")
    fields = (
        "integrated_multimodal_description",
        "overall_soundscape",
        "non_diegetic_music",
    )
    for field in fields:
        assert f"{field}: ..." in text, field
    order = [text.index(f"{field}: ...") for field in fields]
    assert order == sorted(order)
    assert "Never write a reference tag" in text
    assert "<Picture 1>" in text and "<Video 1>" in text and "<Audio 1>" in text
    assert "<d>" in text  # the one angle-bracket tag it still allows
    assert "alignment line" in text
