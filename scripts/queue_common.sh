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
