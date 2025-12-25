ARCH=$(uname -m)

if [ "$ARCH" = "x86_64" ]; then
    PROFILE="amd64"
elif [ "$ARCH" = "aarch64" ]; then
    PROFILE="arm64"
else
    echo "Error: Unsupported architecture: $ARCH"
    exit 1
fi

exec docker compose --profile "$PROFILE" "$@"
