#!/bin/bash
set -euo pipefail

# Ensure HOSTNAME is an exported env var for docker compose
export HOSTNAME="${HOSTNAME:-$(hostname)}"
ARCH=$(uname -m)
DETAIL_ARCH=$(uname -r)
CUDA_MAJOR_VER=$(nvidia-smi | grep -oP 'CUDA Version:\s*\K[0-9]+' | head -n1 || true)
CUDA_MAJOR=$CUDA_MAJOR_VER

if [[ "$ARCH" = "x86_64" ]]; then

    # Optional override:
    #   FORCE_CUDA=12 ./start_interact.sh
    #   FORCE_CUDA=13 ./start_interact.sh
    if [[ "${FORCE_CUDA:-}" == "12" ]]; then
        SERVICE="linux-cuda12-dev"
    elif [[ "${FORCE_CUDA:-}" == "13" ]]; then
        SERVICE="linux-cuda13-dev"
    else
        if (( CUDA_MAJOR >= 13 )); then
            SERVICE="linux-cuda13-dev"
        elif (( CUDA_MAJOR >= 12 )); then
            SERVICE="linux-cuda12-dev"
        else
            echo "ERROR: Detected CUDA ${CUDA_MAJOR}, but this compose file expects >= 12."
            exit 1
        fi
    fi

elif [[ "$ARCH" = "aarch64" ]] && (( CUDA_MAJOR >= 13 )); then
    SERVICE="jetson-thor-dev"

elif [[ "$ARCH" = "aarch64" ]] && (( CUDA_MAJOR >= 12 )); then
    SERVICE="jetson-orin-dev"

else
    echo "Error: Unsupported architecture: $ARCH"
    exit 1
fi

mount_args=()
for path in "$@"; do
    if [[ ! -e "${path}" ]]; then
        echo "ERROR: Mount path does not exist: ${path}"
        exit 1
    fi

    abs_path=$(realpath "${path}")
    mount_args+=("-v" "${abs_path}:${abs_path}")
done

echo "Starting service: ${SERVICE}"

(
    cd ./docker/Linux
    ./compose.sh run --rm --service-ports "${mount_args[@]}" "${SERVICE}" bash
)
