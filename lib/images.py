"""Resolve which vLLM image a workload runs, per hardware platform.

vLLM is published as a separate image per platform, and the tags need not
resemble each other — a release candidate might be `myrepo/vllm:v0.12.0rc2` on
CUDA and `myrepo/vllm-rocm:rc2` on ROCm. So a build can pin one image per
platform, and every workload picks up the image for its GPU's platform
(`platform:` in lib/gpu_profiles.yaml, `cuda` by default):

    VLLM_IMAGE_CUDA    used by H200, B200, ...
    VLLM_IMAGE_ROCM    used by MI300X, MI355X, ...
    VLLM_IMAGE         applies to the platform the image names

`VLLM_COMMIT` is the vLLM revision under test, whichever platform runs it.

Usually one pin is enough, because a platform with no pin of its own is
derivable from the pins the build does have:

  - The release pipeline publishes every platform of a build into one repo,
    tagged `<sha>-<suffix>` (`<sha>-x86_64`, `<sha>-rocm`), so a release
    candidate pinned for one platform names the others.
  - A build testing nightlies resolves the rest to `<repo>:nightly-<commit>`.

A derived ref is a guess, so `resolve` takes an optional `verify` hook
(registry.py) and treats a definitive "no such tag" as reason to skip the
workload. If nothing can be derived either, `resolve` says which variable to
set. Either way the caller skips instead of quietly benchmarking an unrelated
image and reporting it as the build under test.
"""

import os
import re
from typing import NamedTuple

DEFAULT_REPOS = {
    "cuda": "vllm/vllm-openai",
    "rocm": "vllm/vllm-openai-rocm",
}
# Suffix the release pipeline tags each platform's build with, overridable per
# profile with `release_tag_suffix` (an ARM CUDA queue would want aarch64).
RELEASE_TAG_SUFFIX = {
    "cuda": "x86_64",
    "rocm": "rocm",
}


class ResolvedImage(NamedTuple):
    image: str
    commit: str
    platform: str
    # Short (<=70 char, Buildkite's skip-reason limit) explanation of why this
    # platform has no image in this build. Empty when `image` is usable.
    unavailable: str


def _env(name):
    return (os.environ.get(name) or "").strip()


def tag_of(image):
    ref = image.split("@", 1)[0]
    _, sep, tag = ref.rpartition(":")
    return tag if sep and "/" not in tag else ""


def commit_from_image(image):
    """Extract a commit SHA from an image tag, if one is embedded."""
    tag = tag_of(image)
    if not tag:
        return ""
    m = (re.match(r"nightly-([0-9a-f]{7,40})(?:[-_.].*)?$", tag, re.IGNORECASE)
         or re.search(r"(?:^|[-_.])([0-9a-f]{12,40})(?:$|[-_.])", tag, re.IGNORECASE))
    return m.group(1) if m else ""


def platform_of_image(image):
    """The platform an image ref names, so a bare VLLM_IMAGE lands correctly."""
    return "rocm" if "rocm" in image.lower() else "cuda"


def is_nightly(image):
    tag = tag_of(image)
    return tag == "nightly" or tag.startswith("nightly-")


def platform_of_profile(profile):
    platform = str(profile.get("platform") or "cuda").strip().lower()
    if platform not in DEFAULT_REPOS:
        raise ValueError(
            f"unknown platform {platform!r} (expected one of {', '.join(DEFAULT_REPOS)})"
        )
    return platform


def release_sibling(pins, suffix):
    """The pinned release build's tag for another platform, if it has one.

    One release build lands every platform in the same repo as `<sha>-<suffix>`,
    so `…/vllm-release-repo:<sha>-x86_64` implies `…:<sha>-rocm`.
    """
    for pin in pins.values():
        m = re.match(r"([0-9a-f]{40})(?:-|$)", tag_of(pin))
        if m:
            repo = pin.split("@", 1)[0].rpartition(":")[0]
            return f"{repo}:{m.group(1)}-{suffix}"
    return ""


def pinned_images():
    """The images this build pins, keyed by platform."""
    pins = {}
    generic = _env("VLLM_IMAGE")
    if generic:
        pins[platform_of_image(generic)] = generic
    for platform in DEFAULT_REPOS:
        explicit = _env(f"VLLM_IMAGE_{platform.upper()}")
        if explicit:
            pins[platform] = explicit
    return pins


def resolve(vllm, profile, verify=None):
    """Pick the image and vLLM commit for one workload.

    `verify` is an optional ``ref -> True | False | None`` check (see
    registry.py). It is consulted only for refs derived here, never for one
    somebody pinned by hand, and only a definitive False rejects a candidate.
    """
    platform = platform_of_profile(profile)
    slot = platform.upper()
    repo = str(profile.get("image_repo") or "").strip() or DEFAULT_REPOS[platform]
    pins = pinned_images()
    commit = _env("VLLM_COMMIT")

    pinned = pins.get(platform)
    if pinned:
        return ResolvedImage(pinned, commit or commit_from_image(pinned), platform, "")

    nightly_pins = all(is_nightly(i) for i in pins.values())
    if not commit and nightly_pins:
        # A nightly-only build may name its commit solely through its tags.
        commit = next((c for c in map(commit_from_image, pins.values()) if c), "")
    suffix = str(profile.get("release_tag_suffix") or "").strip()
    candidates = []
    sibling = release_sibling(pins, suffix or RELEASE_TAG_SUFFIX[platform])
    if sibling:
        candidates.append(sibling)
    if commit and nightly_pins:
        candidates.append(f"{repo}:nightly-{commit}")
    for candidate in candidates:
        if verify is None or verify(candidate) is not False:
            commit = commit or commit_from_image(candidate)
            return ResolvedImage(candidate, commit, platform, "")

    if candidates:
        commit = commit or commit_from_image(candidates[0])
        return ResolvedImage("", commit, platform, f"no {slot} image built for {commit[:12]}")
    if pins:
        return ResolvedImage(
            "", commit, platform, f"needs a {slot} image: set VLLM_IMAGE_{slot}"
        )

    image = str(vllm.get("image") or f"{repo}:nightly")
    return ResolvedImage(image, commit or commit_from_image(image), platform, "")
