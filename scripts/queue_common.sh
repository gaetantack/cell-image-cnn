# Shared helpers for the queue scripts (source it; run from the repository root).

# skip_if_done <run_id>: true (and prints a skip line) if the run is already in results/experiments.csv.
# Queue scripts call it BEFORE redirecting src.train output to logs/<run_id>.log, so re-running a queue
# does not overwrite the training log of a finished run. Same check as src.train (src.experiments.has_run).
skip_if_done() {
  if python -c "import sys; from src.experiments import has_run; sys.exit(0 if has_run('$1') else 1)"; then
    echo "$(date '+%H:%M') skip $1: already in experiments.csv"
    return 0
  fi
  return 1
}

# Time guard (run 4; replaces the old "minutes since noon" version, which failed after midnight
# and in the daytime). The deadline is a real clock time, fixed ONCE when this file is sourced:
#   - started before 06:30  -> 06:30 today
#   - started at/after 06:30 -> 06:30 tomorrow (an evening start for an overnight run)
# Because it is fixed at start-up, the deadline does not jump to the next day once 06:30 has passed.
# Override for a daytime run:  QUEUE_DEADLINE="2026-11-07 17:00" bash scripts/queue_run4.sh
# Computed in Python, so it works with both BSD (macOS) and GNU date.
DEADLINE_EPOCH=$(python - "$QUEUE_DEADLINE" <<'PY'
import datetime as dt, sys
now = dt.datetime.now()
if sys.argv[1]:                                   # explicit override "YYYY-MM-DD HH:MM"
    deadline = dt.datetime.strptime(sys.argv[1], "%Y-%m-%d %H:%M")
else:
    deadline = now.replace(hour=6, minute=30, second=0, microsecond=0)
    if now >= deadline:                           # evening start -> 06:30 the next morning
        deadline += dt.timedelta(days=1)
print(int(deadline.timestamp()))
PY
)
echo "$(date '+%H:%M') time guard: deadline $(python -c "import datetime as d; print(d.datetime.fromtimestamp($DEADLINE_EPOCH).strftime('%Y-%m-%d %H:%M'))")"

# fits <minutes>: true if a job that takes <minutes> (worst case) and starts now ends before the deadline.
fits() { [ $(( $(date +%s) + $1 * 60 )) -le $DEADLINE_EPOCH ]; }
