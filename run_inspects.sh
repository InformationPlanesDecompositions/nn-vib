#!/bin/sh

set -eu

usage() {
  echo "Usage: $0 [max_parallel]"
  echo "  max_parallel defaults to 4"
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

max_parallel="${1:-4}"
case "$max_parallel" in
  ''|*[!0-9]*)
    echo "Error: max_parallel must be a positive integer" >&2
    exit 1
    ;;
  0)
    echo "Error: max_parallel must be >= 1" >&2
    exit 1
    ;;
esac

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
inspect_py="$script_dir/src/inspect_mlp_ib.py"

if [ ! -f "$inspect_py" ]; then
  echo "Error: expected script at $inspect_py" >&2
  exit 1
fi

failures_file=$(mktemp)
trap 'rm -f "$failures_file"' EXIT INT TERM
export inspect_py failures_file

# Cartesian product of 4 seeds x 6 model specs.
cat <<'EOF' | xargs -n 4 -P "$max_parallel" sh -c '
seed="$1"
hidden1="$2"
z_dim="$3"
hidden2="$4"

printf "Running seed=%s h1=%s z=%s h2=%s\n" "$seed" "$hidden1" "$z_dim" "$hidden2"
if python3 "$inspect_py" --hidden1 "$hidden1" --z_dim "$z_dim" --hidden2 "$hidden2" --seed "$seed"; then
  printf "Done    seed=%s h1=%s z=%s h2=%s\n" "$seed" "$hidden1" "$z_dim" "$hidden2"
else
  printf "FAIL    seed=%s h1=%s z=%s h2=%s\n" "$seed" "$hidden1" "$z_dim" "$hidden2" >&2
  printf "%s %s %s %s\n" "$seed" "$hidden1" "$z_dim" "$hidden2" >>"$failures_file"
  exit 1
fi
' sh
2136623168 386 15 128
2136623168 256 10 64
2136623168 512 10 128
2136623168 386 8 128
2136623168 256 4 64
2136623168 512 4 128
3824702233 386 15 128
3824702233 256 10 64
3824702233 512 10 128
3824702233 386 8 128
3824702233 256 4 64
3824702233 512 4 128
416282721 386 15 128
416282721 256 10 64
416282721 512 10 128
416282721 386 8 128
416282721 256 4 64
416282721 512 4 128
3991408081 386 15 128
3991408081 256 10 64
3991408081 512 10 128
3991408081 386 8 128
3991408081 256 4 64
3991408081 512 4 128
EOF

if [ -s "$failures_file" ]; then
  echo
  echo "Some runs failed:" >&2
  cat "$failures_file" >&2
  exit 1
fi

echo
echo "All inspect runs completed successfully."
