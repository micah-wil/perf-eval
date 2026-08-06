#!/usr/bin/env python3
"""Print a container image's manifest digest as `<repo>@sha256:...`.

Used by lib/server.sh when no Docker daemon is available to ask — the native
(Kubernetes) runtime runs *inside* the image, so there is nothing local to
inspect. This asks the registry what the tag currently points at, which matches
the image in use only if the tag has not moved since the pod pulled it. Prefer
`docker inspect` whenever a daemon exists.

Anonymous pull tokens only; no credentials are read or sent. Prints nothing and
exits non-zero on any failure, since a missing digest is recorded as NULL rather
than failing a run.

Usage: python3 lib/image_digest.py vllm/vllm-openai-rocm:nightly
"""

import json
import os
import sys
import urllib.error
import urllib.request

TIMEOUT = 20
MANIFEST_ACCEPT = ", ".join((
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.docker.distribution.manifest.v2+json",
))
DOCKER_HUB_REGISTRY = "registry-1.docker.io"
DOCKER_HUB_AUTH = "https://auth.docker.io/token"


def split_image(image):
    """Split an image reference into (registry, repository, tag)."""
    ref = image
    if "@" in ref:                      # already digest-pinned
        repo, _, digest = ref.partition("@")
        return None, repo, digest
    host = None
    head, slash, rest = ref.partition("/")
    # A leading component is a registry only if it looks like a host.
    if slash and ("." in head or ":" in head or head == "localhost"):
        host, ref = head, rest
    repo, sep, tag = ref.rpartition(":")
    if not sep or "/" in tag:           # no tag, just a path
        repo, tag = ref, "latest"
    if host is None:
        host = DOCKER_HUB_REGISTRY
        if "/" not in repo:             # official images live under library/
            repo = f"library/{repo}"
    return host, repo, tag


def _get(url, headers, method="GET"):
    req = urllib.request.Request(url, headers=headers, method=method)
    return urllib.request.urlopen(req, timeout=TIMEOUT)


def hub_token(repo):
    url = f"{DOCKER_HUB_AUTH}?service=registry.docker.io&scope=repository:{repo}:pull"
    with _get(url, {}) as resp:
        return json.load(resp).get("token")


def manifest_digest(host, repo, tag):
    headers = {"Accept": MANIFEST_ACCEPT}
    if host == DOCKER_HUB_REGISTRY:
        token = hub_token(repo)
        if token:
            headers["Authorization"] = f"Bearer {token}"
    url = f"https://{host}/v2/{repo}/manifests/{tag}"
    # HEAD is enough: the digest comes back in a header.
    with _get(url, headers, method="HEAD") as resp:
        return resp.headers.get("Docker-Content-Digest")


def main():
    if len(sys.argv) != 2:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    image = sys.argv[1].strip()
    if not image:
        return 2
    try:
        host, repo, tag = split_image(image)
    except Exception:
        return 1
    if host is None:                    # already `repo@sha256:...`
        print(f"{repo}@{tag}")
        return 0
    try:
        digest = manifest_digest(host, repo, tag)
    except (urllib.error.URLError, OSError, ValueError, KeyError) as e:
        print(f"image_digest: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    if not digest:
        print("image_digest: registry returned no Docker-Content-Digest",
              file=sys.stderr)
        return 1
    # Report against the original repo spelling so the value round-trips as a
    # pullable reference.
    base = image.split("@", 1)[0].rsplit(":", 1)[0]
    print(f"{base}@{digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
