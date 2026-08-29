#!/usr/bin/env bash
set -euo pipefail

# Keep parallelism bounded: media tests are memory-heavy and CI/desktop machines
# commonly report many more CPUs than they can safely use for this suite.
if [[ -n "${REFPACK_WORKERS:-}" ]]; then
    workers="$REFPACK_WORKERS"
else
    cpus="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)"
    workers=$((cpus - 1))
fi
if ! [[ "$workers" =~ ^[1-9][0-9]*$ ]]; then
    echo "REFPACK_WORKERS must be a positive integer" >&2
    exit 2
fi
(( workers > 4 )) && workers=4

python=(python)
if [[ -x ".venv/bin/python" ]]; then
    python=(.venv/bin/python)
fi

run_fast() {
    "${python[@]}" -m pytest -m 'not slow' -n "$workers" "$@"
}

case "${1:-fast}" in
    fast)
        run_fast
        ;;
    full)
        run_fast
        # RSS/real-media tests must never overlap one another or the fast workers.
        "${python[@]}" -m pytest -m slow -n 0
        ;;
    changed)
        mapfile -t changed_files < <(
            {
                git diff --name-only HEAD
                git ls-files --others --exclude-standard
            } | sort -u
        )
        non_test_change=0
        test_files=()
        for path in "${changed_files[@]}"; do
            if [[ "$path" == tests/*.py ]]; then
                test_files+=("$path")
            elif [[ -n "$path" ]]; then
                non_test_change=1
            fi
        done
        if (( non_test_change || ${#test_files[@]} == 0 )); then
            run_fast
        else
            run_fast "${test_files[@]}"
        fi
        ;;
    *)
        echo "usage: $0 {changed|fast|full}" >&2
        exit 2
        ;;
esac
