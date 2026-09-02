"""Small, dependency-light helpers for reproducible inference metadata."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, is_dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Mapping, Sequence


_PACKAGE_DISTRIBUTIONS = {
    "accelerate": "accelerate",
    "flash_attn": "flash-attn",
    "huggingface_hub": "huggingface-hub",
    "kenlm": "kenlm",
    "kaldi_native_fbank": "kaldi-native-fbank",
    "librosa": "librosa",
    "nemo_toolkit": "nemo-toolkit",
    "numpy": "numpy",
    "omegaconf": "omegaconf",
    "onnx": "onnx",
    "onnx_asr": "onnx-asr",
    "onnxruntime": "onnxruntime",
    "peft": "peft",
    "pyctcdecode": "pyctcdecode",
    "qwen_omni_utils": "qwen-omni-utils",
    "safetensors": "safetensors",
    "sherpa_onnx": "sherpa-onnx",
    "soundfile": "soundfile",
    "torch": "torch",
    "torchaudio": "torchaudio",
    "transformers": "transformers",
    "yaml": "PyYAML",
}

# The BookBot matrix runner verifies the selected files against the tracked
# SHA-256 manifest associated with these immutable source revisions.
_HUGGING_FACE_SOURCES = {
    "sherpa-onnx-ort-zipformer-streaming-robust-sw-v4": (
        "bookbot/sherpa-onnx-ort-zipformer-streaming-robust-sw-v4",
        "311c41c8770242c02478d4569fcf5e0cd00c1218",
    ),
    "sherpa-onnx-zipformer-streaming-robust-sw-v4": (
        "bookbot/sherpa-onnx-zipformer-streaming-robust-sw-v4",
        "0e52da6c03294fd983f3a8621b32ac6a71b4787d",
    ),
    "zipformer-streaming-robust-sw-v4": (
        "bookbot/zipformer-streaming-robust-sw-v4",
        "f27bc1620ac08c6a4bc6a1cf6072d592e7b09a49",
    ),
    "gemma-4-E2B-it": (
        "google/gemma-4-E2B-it",
        "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
    ),
    "hubert-large-ls960-ft": (
        "facebook/hubert-large-ls960-ft",
        "ece5fabbf034c1073acae96d5401b25be96709d8",
    ),
    "mms-1b-all": (
        "facebook/mms-1b-all",
        "3d33597edbdaaba14a8e858e2c8caa76e3cec0cd",
    ),
    "paza-Phi-4-multimodal-instruct": (
        "microsoft/paza-Phi-4-multimodal-instruct",
        "1e78f4f84fd8a92f132295a32e5cb71b5321dcab",
    ),
    "paza-whisper-large-v3-turbo": (
        "microsoft/paza-whisper-large-v3-turbo",
        "8a0bd24795b298b2f3678fcd8129c0a1c2b4ee22",
    ),
    "Qwen2.5-Omni-7B": (
        "Qwen/Qwen2.5-Omni-7B",
        "ae9e1690543ffd5c0221dc27f79834d0294cba00",
    ),
    "w2v-bert-2.0-swahili-asr": (
        "badrex/w2v-bert-2.0-swahili-asr",
        "10e85418ae5978a084c06de2c47448acb1a4e0c8",
    ),
    "wav2vec2-xls-r-300m-swahili-cv-fleurs-alffa-alphabets-phonemes-bookbot": (
        "bookbot/wav2vec2-xls-r-300m-swahili-cv-fleurs-alffa-alphabets-phonemes-bookbot",
        "b83c1c4f1da4eef4a3a7b8c8c32d7d1a91bc0011",
    ),
    "wav2vec2-xls-r-300m-swahili-cv-fleurs-alffa-word-lm": (
        "bookbot/wav2vec2-xls-r-300m-swahili-cv-fleurs-alffa-word-lm",
        "f71831f7a9ad8d0ab1fa16ff071985472a191007",
    ),
    "whisper-large": (
        "openai/whisper-large",
        "4ef9b41f0d4fe232daafdb5f76bb1dd8b23e01d7",
    ),
    "whisper-large-v2": (
        "openai/whisper-large-v2",
        "ae4642769ce2ad8fc292556ccea8e901f1530655",
    ),
}

_CONFIG_IDENTITY_FILES = (
    "alphabet.json",
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "language_model/attrs.json",
    "preprocessor_config.json",
    "processor_config.json",
    "required_operators.config",
    "tokenizer_config.json",
    "tokens.txt",
    "vocab.json",
)

_LOCAL_ARTIFACT_SUFFIXES = {
    ".bin",
    ".int8",
    ".model",
    ".nemo",
    ".onnx",
    ".ort",
    ".pt",
    ".safetensors",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    identity: dict[str, Any] = {"path": str(resolved), "exists": resolved.is_file()}
    if resolved.is_file():
        identity.update(
            size_bytes=resolved.stat().st_size,
            sha256=sha256_file(resolved),
        )
    return identity


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_download_manifests(model_path: Path, artifact: str) -> dict[str, Any]:
    parent = model_path.parent
    source_path = parent / f".{artifact}.download-source.tsv"
    sizes_path = parent / f".{artifact}.download-sizes.tsv"
    completion_path = parent / f".{artifact}.download-complete"
    payload: dict[str, Any] = {"completion_marker": completion_path.is_file()}

    if source_path.is_file():
        source_values: dict[str, str] = {}
        for raw_line in source_path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            key, separator, value = raw_line.partition("\t")
            if separator:
                source_values[key] = value
        payload["source"] = {
            **file_identity(source_path),
            "values": source_values,
        }

    if sizes_path.is_file():
        file_count = 0
        total_size_bytes = 0
        malformed_lines = 0
        for raw_line in sizes_path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            size, separator, _relative_path = raw_line.partition("\t")
            if not separator or not size.isdigit():
                malformed_lines += 1
                continue
            file_count += 1
            total_size_bytes += int(size)
        payload["sizes"] = {
            **file_identity(sizes_path),
            "file_count": file_count,
            "total_size_bytes": total_size_bytes,
            "malformed_lines": malformed_lines,
        }
    return payload


def _read_hugging_face_identity(model_path: Path) -> dict[str, Any]:
    metadata_root = model_path / ".cache" / "huggingface" / "download"
    revisions: set[str] = set()
    records: dict[str, Any] = {}
    if metadata_root.is_dir():
        for metadata_path in metadata_root.rglob("*.metadata"):
            try:
                lines = metadata_path.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
            except OSError:
                continue
            if not lines:
                continue
            revision = lines[0].strip().lower()
            if len(revision) == 40 and all(
                character in "0123456789abcdef" for character in revision
            ):
                revisions.add(revision)
            relative_name = metadata_path.relative_to(metadata_root).as_posix()
            relative_name = relative_name.removesuffix(".metadata")
            record: dict[str, Any] = {"revision": revision}
            if len(lines) > 1:
                record["etag"] = lines[1].strip().strip('"')
            downloaded_file = model_path / relative_name
            if downloaded_file.is_file():
                record["size_bytes"] = downloaded_file.stat().st_size
            records[relative_name] = record
    return {
        "metadata_files": len(records),
        "revisions": sorted(revisions),
        "consistent_revision": next(iter(revisions)) if len(revisions) == 1 else None,
        "etag_manifest_sha256": canonical_json_sha256(records),
        "files": records,
    }


def model_artifact_identity(
    model_path: str | Path,
    *,
    artifact: str,
    selected_files: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Describe the selected local model without re-hashing covered Hub weights."""
    path = Path(model_path)
    identity: dict[str, Any] = {
        "artifact": artifact,
        "path": str(path),
        "exists": path.exists(),
        "kind": "file"
        if path.is_file()
        else "directory"
        if path.is_dir()
        else "missing",
    }
    source = _HUGGING_FACE_SOURCES.get(artifact)
    if source is not None:
        identity["intended_source"] = {
            "repository": source[0],
            "revision": source[1],
        }

    download_manifests = _read_download_manifests(path, artifact)
    if (
        download_manifests["completion_marker"]
        or "source" in download_manifests
        or "sizes" in download_manifests
    ):
        identity["download_manifests"] = download_manifests

    if path.is_file():
        identity.update(
            size_bytes=path.stat().st_size,
            sha256=sha256_file(path),
        )
        return identity
    if not path.is_dir():
        return identity

    local_hf = _read_hugging_face_identity(path)
    if local_hf["metadata_files"]:
        identity["local_hugging_face_metadata"] = local_hf

    configuration_files: dict[str, Any] = {}
    for relative_name in _CONFIG_IDENTITY_FILES:
        candidate = path / relative_name
        if candidate.is_file():
            configuration_files[relative_name] = {
                "size_bytes": candidate.stat().st_size,
                "sha256": sha256_file(candidate),
            }
    if configuration_files:
        identity["configuration_files"] = configuration_files

    # A Hub ETag identifies files represented in the local download metadata.
    # Hash only executable artifacts that are not represented there. This keeps
    # complete Hub snapshots cheap while binding partial snapshots and local
    # exports to the actual files used by inference.
    hub_files = set(local_hf["files"])
    artifact_files: dict[str, Any] = {}
    if selected_files is None:
        candidates = sorted(item for item in path.rglob("*") if item.is_file())
    else:
        normalized_selected_files: list[str] = []
        candidates = []
        for raw_relative_name in selected_files:
            relative = Path(raw_relative_name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(
                    "Selected artifact files must be relative to the model directory: "
                    f"{raw_relative_name}"
                )
            normalized_name = relative.as_posix()
            normalized_selected_files.append(normalized_name)
            candidate = path / relative
            if candidate.is_file():
                candidates.append(candidate)
        identity["selected_files"] = normalized_selected_files

    for candidate in candidates:
        relative = candidate.relative_to(path)
        relative_name = relative.as_posix()
        if any(part in {".cache", ".git"} for part in relative.parts):
            continue
        if relative_name in _CONFIG_IDENTITY_FILES:
            continue
        if (
            selected_files is None
            and candidate.suffix.lower() not in _LOCAL_ARTIFACT_SUFFIXES
        ):
            continue
        if relative_name in hub_files:
            continue
        artifact_files[relative_name] = {
            "size_bytes": candidate.stat().st_size,
            "sha256": sha256_file(candidate),
        }
    if artifact_files:
        identity["artifact_files"] = artifact_files
        identity["artifact_manifest_sha256"] = canonical_json_sha256(artifact_files)
    return identity


def _find_repository_root(start: Path) -> Path:
    resolved = start.resolve()
    for candidate in (resolved, *resolved.parents):
        if (candidate / ".git").exists():
            return candidate
    return resolved


def _source_tree_identity(root: Path) -> dict[str, Any]:
    candidates: set[Path] = set()
    for name in (
        "docker-compose.yml",
        "infer.py",
        "requirements.txt",
        "requirements-dev.txt",
    ):
        candidate = root / name
        if candidate.is_file():
            candidates.add(candidate)
    candidates.update(path for path in root.glob("run_*inference.sh") if path.is_file())

    docker_root = root / "docker"
    if docker_root.is_dir():
        candidates.update(path for path in docker_root.rglob("*") if path.is_file())

    inference_root = root / "inference"
    if inference_root.is_dir():
        for path in inference_root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(inference_root)
            if any(part in {"__pycache__", "models", "tmp"} for part in relative.parts):
                continue
            if path.suffix.lower() in {".json", ".py", ".sh", ".yaml", ".yml"}:
                candidates.add(path)

    files: dict[str, str] = {}
    for candidate in sorted(candidates):
        try:
            files[candidate.relative_to(root).as_posix()] = sha256_file(candidate)
        except OSError:
            continue
    return {
        "file_count": len(files),
        "sha256": canonical_json_sha256(files),
        "files": files,
    }


def _git_files_identity(root: Path) -> dict[str, Any]:
    marker = root / ".git"
    git_directory = marker
    try:
        if marker.is_file():
            marker_value = marker.read_text(encoding="utf-8").strip()
            prefix, separator, value = marker_value.partition(":")
            if separator and prefix.lower() == "gitdir":
                git_directory = (root / value.strip()).resolve()
        head_value = (git_directory / "HEAD").read_text(encoding="utf-8").strip()
    except OSError as exc:
        return {
            "available": False,
            "root": str(root),
            "source": "git_files_fallback",
            "error": str(exc),
        }

    branch: str | None = None
    commit: str | None = None
    if head_value.startswith("ref: "):
        reference = head_value.removeprefix("ref: ").strip()
        if reference.startswith("refs/heads/"):
            branch = reference.removeprefix("refs/heads/")
        reference_path = git_directory / reference
        try:
            commit = reference_path.read_text(encoding="utf-8").strip()
        except OSError:
            packed_refs = git_directory / "packed-refs"
            try:
                for raw_line in packed_refs.read_text(encoding="utf-8").splitlines():
                    if raw_line.startswith(("#", "^")):
                        continue
                    candidate_commit, separator, candidate_ref = raw_line.partition(" ")
                    if separator and candidate_ref == reference:
                        commit = candidate_commit
                        break
            except OSError:
                pass
    else:
        commit = head_value

    return {
        "available": bool(commit),
        "root": str(root),
        "commit": commit,
        "branch": branch,
        "dirty": None,
        "status_porcelain": None,
        "source": "git_files_fallback",
        "dirty_state_note": "Git executable unavailable; source_tree identifies exact inference code.",
    }


def git_identity(start: str | Path | None = None) -> dict[str, Any]:
    command_root = _find_repository_root(Path(start or Path.cwd()))

    def run(*arguments: str) -> str:
        return subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={command_root}",
                "-C",
                str(command_root),
                *arguments,
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    source_tree = _source_tree_identity(command_root)
    try:
        root = run("rev-parse", "--show-toplevel")
        commit = run("rev-parse", "HEAD")
        branch_value = run("rev-parse", "--abbrev-ref", "HEAD")
        status = run("status", "--porcelain", "--untracked-files=normal")
    except (OSError, subprocess.CalledProcessError) as exc:
        fallback = _git_files_identity(command_root)
        fallback["git_command_error"] = str(exc)
        fallback["source_tree"] = source_tree
        return fallback
    return {
        "available": True,
        "root": root,
        "commit": commit,
        "branch": None if branch_value == "HEAD" else branch_value,
        "dirty": bool(status),
        "status_porcelain": status.splitlines(),
        "source": "git_command",
        "source_tree": source_tree,
    }


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for output_name, distribution in _PACKAGE_DISTRIBUTIONS.items():
        try:
            versions[output_name] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            continue
    return versions


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return json_safe(to_dict())
        except Exception:
            return {}
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            return json_safe(OmegaConf.to_container(value, resolve=True))
    except (ImportError, TypeError, ValueError):
        pass
    if hasattr(value, "__dict__"):
        return json_safe(vars(value))
    return str(value)


def generation_provenance(
    generation_config: Any,
    *,
    requested_kwargs: Mapping[str, Any],
    call_kwargs: Mapping[str, Any],
    conditional_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    defaults = json_safe(generation_config)
    if not isinstance(defaults, dict):
        defaults = {}
    actual_call_kwargs = json_safe(dict(call_kwargs))
    effective = dict(defaults)
    effective.update(actual_call_kwargs)
    return {
        "model_generation_config": defaults,
        "requested_profile_kwargs": json_safe(dict(requested_kwargs)),
        "actual_call_kwargs": actual_call_kwargs,
        "effective_generation_config": effective,
        "conditional_call_overrides": json_safe(dict(conditional_overrides or {})),
    }
