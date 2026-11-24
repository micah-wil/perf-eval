#!/usr/bin/env bash
# Throwaway probe: what Docker access a GPU pod has. Never fails the step.
# Prints env var names only, not values, since they can hold credentials.
set +e
echo "--- docker socket"
ls -l /var/run/docker.sock /run/docker.sock 2>&1
echo "--- docker cli"
command -v docker docker-compose 2>&1 || echo "no docker cli"
timeout 20 docker version 2>&1
timeout 20 docker compose version 2>&1
echo "--- env (names only)"
env | grep -i -E 'docker|container' | cut -d= -f1
echo "--- capabilities"
grep -E 'Cap(Eff|Bnd)' /proc/self/status
command -v capsh >/dev/null && capsh --decode="$(awk '/CapEff/ {print $2}' /proc/self/status)"
echo "--- identity and network"
id
hostname
hostname -i 2>&1
cat /proc/1/cgroup 2>&1 | head -3
mount | grep -i -E 'docker|containerd' 2>&1
exit 0
