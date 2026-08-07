#!/usr/bin/env python3
"""Stdlib-only regression tests for generate_pipeline.py.

Run with ``python3 .buildkite/test_generate_pipeline.py`` (needs only pyyaml,
which the pipeline already installs). No pytest / GPU / network required.
Image resolution itself is covered by ``python3 lib/test_images.py``.

Guards two things. The HF-cache volume behaviour: the AMD k8s plugin must NOT
emit a root-disk hostPath by default (that leaked model caches onto node root
disks), and the volume source must be overridable per-cluster. And how a step
carries its image: every step is pinned to the image its platform resolved to,
and a platform with no image in this build produces a skipped step.
"""

import contextlib
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "generate_pipeline", os.path.join(HERE, "generate_pipeline.py")
)
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)

IMAGE_VARS = ("VLLM_IMAGE", "VLLM_IMAGE_CUDA", "VLLM_IMAGE_ROCM", "VLLM_COMMIT")


@contextlib.contextmanager
def build_env(**pins):
    saved = {k: os.environ.pop(k, None) for k in IMAGE_VARS}
    os.environ.update(pins)
    try:
        yield
    finally:
        for k in IMAGE_VARS:
            os.environ.pop(k, None)
        os.environ.update({k: v for k, v in saved.items() if v is not None})


def _step(gpu, **extra):
    data = {"name": f"test-{gpu.lower()}", "gpu": gpu, "num_gpus": 8,
            "vllm": {"model": "test/model"}, "vllm_bench": {"configs": []}, **extra}
    return g.make_step(f"workloads/test_{gpu.lower()}.yaml", data, g.load_profiles())


def _amd_volumes(profile, gpu="MI300X"):
    plugin = g.amd_k8s_plugin("img", 8, profile, gpu)
    patch = plugin["kubernetes"]["podSpecPatch"]
    vols = {v["name"]: v for v in patch["volumes"]}
    return patch, vols


def test_default_hf_cache_is_emptydir_not_hostpath():
    """Default (no override, no profile field) must be an emptyDir, never a
    hostPath — a hostPath on an unmounted node path is what filled root disks."""
    _, vols = _amd_volumes({})
    hf = vols["hf-cache"]
    assert "emptyDir" in hf, f"expected emptyDir default, got {hf}"
    assert "hostPath" not in hf, f"default must not be a hostPath: {hf}"


def test_hf_home_mount_matches_env():
    """Whatever the volume source, the mount path and HF_HOME must agree so vLLM
    finds its cache at the advertised location."""
    patch, vols = _amd_volumes({"hf_home": "/root/.cache/huggingface"})
    c = patch["containers"][0]
    mount = next(m for m in c["volumeMounts"] if m["name"] == "hf-cache")
    hf_home = next(e for e in c["env"] if e["name"] == "HF_HOME")
    assert mount["mountPath"] == hf_home["value"] == "/root/.cache/huggingface"


def test_profile_field_overrides_source():
    """A profile-level hf_cache_volume sets the source but keeps the volume name."""
    pvc = {"persistentVolumeClaim": {"claimName": "hf-cache-pvc"}}
    _, vols = _amd_volumes({"hf_cache_volume": pvc})
    assert vols["hf-cache"] == {"name": "hf-cache", **pvc}


def test_env_override_wins_over_profile_and_default():
    """{GPU}_HF_CACHE_VOLUME (per-cluster) overrides the profile and default."""
    key = "MI355X_HF_CACHE_VOLUME"
    prev = os.environ.get(key)
    os.environ[key] = '{"persistentVolumeClaim":{"claimName":"buildkite-hf-cache"}}'
    try:
        _, vols = _amd_volumes(
            {"hf_cache_volume": {"emptyDir": {}}}, gpu="MI355X"
        )
    finally:
        if prev is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prev
    assert vols["hf-cache"] == {
        "name": "hf-cache",
        "persistentVolumeClaim": {"claimName": "buildkite-hf-cache"},
    }


def test_env_override_is_scoped_per_gpu():
    """An override for one GPU key must not leak into another GPU's volume."""
    key = "MI300X_HF_CACHE_VOLUME"
    prev = os.environ.get(key)
    os.environ[key] = '{"persistentVolumeClaim":{"claimName":"only-mi300"}}'
    try:
        _, vols = _amd_volumes({}, gpu="MI355X")
    finally:
        if prev is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prev
    assert "emptyDir" in vols["hf-cache"], vols["hf-cache"]


def test_shipped_amd_profiles_have_no_rootdisk_hostpath():
    """The committed MI300X/MI355X profiles must not reintroduce an hf_home under
    /mnt/shared (the concrete path that leaked onto root disks)."""
    profiles = g.load_profiles()
    for gpu in ("MI300X", "MI355X"):
        hf_home = (profiles.get(gpu) or {}).get("hf_home") or ""
        assert not hf_home.startswith("/mnt/shared"), (
            f"{gpu} hf_home={hf_home!r} would land on the node root disk"
        )


def test_step_is_pinned_to_its_platform_image():
    """A ROCm step runs the ROCm pin — in its pod spec and in the env the job
    re-reads — even when the CUDA pin's name looks nothing like it."""
    with build_env(
        VLLM_IMAGE_CUDA="myrepo/vllm:v0.12.0rc2",
        VLLM_IMAGE_ROCM="other.registry/amd-vllm:rc2-final",
        VLLM_COMMIT="abc1234def5678",
    ):
        amd, nvidia = _step("MI355X"), _step("H200")
    pod_image = amd["plugins"][0]["kubernetes"]["podSpecPatch"]["containers"][0]["image"]
    assert pod_image == "other.registry/amd-vllm:rc2-final", pod_image
    assert amd["env"]["VLLM_IMAGE_ROCM"] == "other.registry/amd-vllm:rc2-final"
    assert nvidia["env"]["VLLM_IMAGE_CUDA"] == "myrepo/vllm:v0.12.0rc2"
    assert "VLLM_IMAGE_ROCM" not in nvidia["env"]


def test_step_is_skipped_when_its_platform_has_no_image():
    """A CUDA-only release candidate leaves AMD workloads visible-but-skipped
    rather than running them against an unrelated image."""
    with build_env(VLLM_IMAGE="myrepo/vllm:v0.12.0rc2", VLLM_COMMIT="abc1234def5678"):
        amd = _step("MI355X")
    assert amd["skip"], amd
    assert "VLLM_IMAGE_ROCM" in amd["skip"]
    assert len(amd["skip"]) <= 70, "Buildkite caps skip reasons at 70 chars"
    assert "plugins" not in amd, "a skipped step must not name an image to pull"
    assert "VLLM_IMAGE_ROCM" not in (amd.get("env") or {})


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"ok   {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
