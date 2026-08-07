"""Ask a container registry whether an image tag exists.

Used at pipeline-generation time, and only for refs perf-eval derived itself
(see images.py) — an image somebody pinned by hand is scheduled as given. The
point is to catch the build that published one platform but not another before
it costs a GPU job, and to say so in the skipped step instead of failing on an
image pull an hour into the queue.

`image_exists` answers True, False, or None for "couldn't tell". Only a
registry saying *not found* is a False; a network error, a rate limit, or a
registry that needs credentials we don't have is None, so a flaky check can
never quietly drop coverage.
"""

import functools
import json
import re
import urllib.error
import urllib.parse
import urllib.request

DOCKER_HUB = "registry-1.docker.io"
MANIFEST_TYPES = ", ".join((
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.docker.distribution.manifest.v2+json",
))
TIMEOUT_S = 10


def parse_ref(ref):
    """Split an image ref into (registry host, repository, tag)."""
    name, _, tag = ref.split("@", 1)[0].rpartition(":")
    if not name or "/" in tag:
        name, tag = ref, "latest"
    host, slash, path = name.partition("/")
    if slash and ("." in host or ":" in host or host == "localhost"):
        return host, path, tag
    return DOCKER_HUB, name if slash else f"library/{name}", tag


def _manifest_request(url, token=None):
    request = urllib.request.Request(url, method="HEAD")
    request.add_header("Accept", MANIFEST_TYPES)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    return urllib.request.urlopen(request, timeout=TIMEOUT_S)


def _anonymous_token(challenge, repo):
    """Answer a 401's Bearer challenge with an anonymous pull token."""
    fields = dict(re.findall(r'(\w+)="([^"]*)"', challenge))
    realm = fields.get("realm")
    if not realm:
        return ""
    query = {"scope": f"repository:{repo}:pull"}
    if fields.get("service"):
        query["service"] = fields["service"]
    with urllib.request.urlopen(
        f"{realm}?{urllib.parse.urlencode(query)}", timeout=TIMEOUT_S
    ) as response:
        body = json.load(response)
    return body.get("token") or body.get("access_token") or ""


@functools.lru_cache(maxsize=None)
def image_exists(ref):
    """True if the tag is in the registry, False if it isn't, None if unknown."""
    host, repo, tag = parse_ref(ref)
    url = f"https://{host}/v2/{repo}/manifests/{tag}"
    try:
        try:
            _manifest_request(url).close()
            return True
        except urllib.error.HTTPError as e:
            if e.code != 401:
                raise
            token = _anonymous_token(e.headers.get("WWW-Authenticate") or "", repo)
            if not token:
                return None
            _manifest_request(url, token).close()
            return True
    except urllib.error.HTTPError as e:
        return False if e.code == 404 else None
    except (OSError, ValueError):
        return None
