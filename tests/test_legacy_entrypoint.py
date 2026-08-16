from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import infer
import pytest


def test_existing_model_path_is_not_remapped(tmp_path: Path) -> None:
    model = tmp_path / "existing.nemo"
    model.write_bytes(b"model")

    assert infer.remap_legacy_model_path(str(model)) == str(model)


@pytest.mark.parametrize(
    "legacy_path",
    [
        "nemo_inference/models/nested/example.nemo",
        r"nemo_inference\models\nested\example.nemo",
    ],
)
def test_old_nemo_model_prefix_is_remapped(legacy_path: str) -> None:
    remapped = Path(
        infer.remap_legacy_model_path(legacy_path)
    )

    assert remapped == (
        Path(infer.__file__).resolve().parent
        / "inference"
        / "nemo"
        / "models"
        / "nested"
        / "example.nemo"
    )


def test_new_or_unrelated_missing_model_path_is_not_remapped() -> None:
    new_path = "inference/nemo/models/example.nemo"
    unrelated = "somewhere/example.nemo"

    assert infer.remap_legacy_model_path(new_path) == new_path
    assert infer.remap_legacy_model_path(unrelated) == unrelated


def test_root_main_delegates_to_legacy_api_with_mapper(monkeypatch) -> None:
    calls: dict[str, object] = {}
    module = ModuleType("inference.nemo.infer")

    def fake_legacy_main(argv, model_path_mapper):
        calls["argv"] = argv
        calls["mapper"] = model_path_mapper

    module.legacy_main = fake_legacy_main  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "inference.nemo.infer", module)

    argv = ["--model", "inference/nemo/models/example.nemo"]
    infer.main(argv)

    assert calls == {"argv": argv, "mapper": infer.remap_legacy_model_path}
