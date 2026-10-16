#!/bin/bash
# Root-level entry point kept for the server-setup hint; the launcher itself
# (promoted h6s1 recipe) lives in scripts/bash/train_prefilter.sh.
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/bash/train_prefilter.sh" "$@"
