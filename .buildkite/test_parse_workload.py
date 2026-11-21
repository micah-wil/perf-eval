import importlib.util
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "parse_workload", ROOT / "lib" / "parse_workload.py"
)
assert SPEC and SPEC.loader
parse_workload = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(parse_workload)

IMAGE_VARS = ("VLLM_IMAGE", "VLLM_IMAGE_CUDA", "VLLM_IMAGE_ROCM", "VLLM_COMMIT")


@pytest.fixture(autouse=True)
def build_env(monkeypatch):
    """Resolve images from what the test sets, not what the build's env pins."""
    for var in IMAGE_VARS:
        monkeypatch.delenv(var, raising=False)


def write_workload(tmp_path: Path, startup_timeout_s: object = 3600, **extra) -> Path:
    (tmp_path / "lib").mkdir()
    (tmp_path / "workloads").mkdir()
    (tmp_path / "lib" / "gpu_profiles.yaml").write_text(yaml.safe_dump({"H200": {}}))
    workload = tmp_path / "workloads" / "workload.yaml"
    workload.write_text(
        yaml.safe_dump(
            {
                **extra,
                "name": "test",
                "gpu": "H200",
                "vllm": {
                    "model": "test/model",
                    "startup_timeout_s": startup_timeout_s,
                    "serve_args": "--tensor-parallel-size 8",
                },
                "vllm_bench": {
                    "configs": [
                        {
                            "name": "smoke",
                            "input_len": 1,
                            "output_len": 1,
                            "num_prompts": 1,
                            "max_concurrency": 1,
                        }
                    ]
                },
            }
        )
    )
    return workload


def test_emits_server_startup_timeout(tmp_path, capsys):
    parse_workload.main(str(write_workload(tmp_path, 7200)))

    assert "WORKLOAD_SERVER_STARTUP_TIMEOUT=7200" in capsys.readouterr().out


@pytest.mark.parametrize("value", [0, -1, True, "7200"])
def test_rejects_invalid_server_startup_timeout(tmp_path, value):
    with pytest.raises(SystemExit, match="must be a positive integer"):
        parse_workload.main(str(write_workload(tmp_path, value)))


def test_cuda_release_image_does_not_select_rocm_commit(monkeypatch):
    """A CUDA release image embeds a commit, but AMD has no build of it: fall
    back to the ROCm nightly rather than a same-commit image that never existed.
    """
    monkeypatch.setenv(
        "VLLM_IMAGE", "public.ecr.aws/example/release:abc123def456-x86_64"
    )

    image, commit = parse_workload.resolve_image(
        {}, {"image_repo": "vllm/vllm-openai-rocm"}
    )

    assert image == "vllm/vllm-openai-rocm:nightly"
    assert commit == ""


@pytest.mark.parametrize(
    ("profile", "env", "expected"),
    [
        ({"nsys": True}, None, True),
        ({}, None, False),
        ({"nsys": True}, "0", False),
        ({}, "1", True),
        ({"nsys": True, "image_repo": "vllm/vllm-openai-rocm"}, "1", False),
    ],
)
def test_nsys_follows_profile_with_build_override(monkeypatch, profile, env, expected):
    if env is None:
        monkeypatch.delenv("NSYS_PROFILE", raising=False)
    else:
        monkeypatch.setenv("NSYS_PROFILE", env)

    assert parse_workload.nsys_enabled(profile) is expected


FASTSAFETENSORS = {"--load-format": "fastsafetensors"}


@pytest.mark.parametrize(
    ("serve_args", "defaults", "expected"),
    [
        ("--tensor-parallel-size 8", FASTSAFETENSORS,
         "--tensor-parallel-size 8 --load-format fastsafetensors"),
        ("--load-format auto", FASTSAFETENSORS, "--load-format auto"),
        ("--load-format=dummy", FASTSAFETENSORS, "--load-format=dummy"),
        ("", {"--enforce-eager": True}, "--enforce-eager"),
        ("--max-num-seqs 8", None, "--max-num-seqs 8"),
    ],
)
def test_default_serve_args_apply_unless_the_workload_sets_them(
    serve_args, defaults, expected
):
    assert parse_workload.apply_default_serve_args(serve_args, defaults) == expected


KVV = {
    "serve_args": "--reasoning-parser kimi_k3",
    "jobs": ["verify_k3_features", "ocrbench_thinking_effort:max"],
}


@pytest.mark.parametrize(("env", "enabled"), [(None, False), ("0", False), ("1", True)])
def test_kvv_runs_only_when_the_build_opts_in(tmp_path, capsys, monkeypatch, env, enabled):
    if env is None:
        monkeypatch.delenv("RUN_KVV", raising=False)
    else:
        monkeypatch.setenv("RUN_KVV", env)

    parse_workload.main(str(write_workload(tmp_path, kvv=KVV)))
    out = capsys.readouterr().out

    if enabled:
        assert "WORKLOAD_KVV_JOBS='verify_k3_features ocrbench_thinking_effort:max'" in out
        assert "WORKLOAD_SERVE_ARGS='--tensor-parallel-size 8 --reasoning-parser kimi_k3'" in out
    else:
        assert "WORKLOAD_KVV_JOBS=''" in out
        assert "WORKLOAD_SERVE_ARGS='--tensor-parallel-size 8'" in out


@pytest.mark.parametrize(
    "kvv",
    [{"jobs": []}, {"jobs": ["a b"]}, {"jobs": ["a"], "bogus": 1},
     {"jobs": ["a"], "extra_timeout_in_minutes": -1}],
)
def test_rejects_invalid_kvv_block(tmp_path, monkeypatch, kvv):
    monkeypatch.delenv("RUN_KVV", raising=False)
    with pytest.raises(SystemExit, match="kvv"):
        parse_workload.main(str(write_workload(tmp_path, kvv=kvv)))
