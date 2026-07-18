#!/usr/bin/env bash
# Generic multi-node torchrun launcher.
#
# This file intentionally contains no cluster-specific paths or experiment
# hyperparameters. A tracked recipe must define the required variables and
# arrays, then source this file and call launch_multinode.
set -euo pipefail

_shell_quote() {
  printf '%q' "$1"
}

_append_word() {
  local -n _dst=$1
  _dst+=" $(_shell_quote "$2")"
}

_require_scalar() {
  local name=$1
  if [[ -z ${!name:-} ]]; then
    echo "ERROR: required variable $name is not set" >&2
    return 2
  fi
}

_write_launch_manifest() {
  local manifest_dir=$1
  local timestamp=$2
  local manifest="$manifest_dir/launch-${timestamp}.txt"
  local command_file="$manifest_dir/launch-${timestamp}.cmd"

  mkdir -p "$manifest_dir"
  {
    echo "timestamp_utc=$timestamp"
    echo "recipe=${RECIPE_FILE:-unknown}"
    echo "code_revision=$CODE_REV"
    if [[ -n ${ORIGINAL_RUN_CODE_REV:-} ]]; then
      echo "original_run_code_revision=$ORIGINAL_RUN_CODE_REV"
    fi
    echo "repo_url=$REPO_URL"
    echo "local_repo=$LOCAL_REPO"
    if [[ -n ${VENV_DIR:-} ]]; then
      echo "venv_dir=$VENV_DIR"
    else
      echo "conda_env=$CONDA_ENV"
    fi
    echo "nnodes=${#NODES[@]}"
    echo "nodes=${NODES[*]}"
    echo "log_dir=$LOGDIR"
    echo "output_dir=$OUT"
    echo "code_sync_mode=${CODE_SYNC_MODE:-verify}"
    printf 'remote_env='; printf '%q ' "${REMOTE_ENV[@]}"; echo
    printf 'torchrun_common_args='; printf '%q ' "${TORCHRUN_COMMON_ARGS[@]}"; echo
    printf 'train_args='; printf '%q ' "${TRAIN_ARGS[@]}"; echo
  } > "$manifest"

  {
    printf 'torchrun '
    printf '%q ' "${TORCHRUN_COMMON_ARGS[@]}"
    printf '%s ' '--node_rank=${NODE_RANK}'
    printf '%q ' "${TRAIN_ARGS[@]}"
    echo
  } > "$command_file"

  if [[ -n ${RECIPE_FILE:-} && -f ${RECIPE_FILE:-} ]]; then
    cp "$RECIPE_FILE" "$manifest_dir/launch-${timestamp}.recipe.sh"
  fi

  cp "$manifest" "$manifest_dir/latest.txt"
  echo "launch manifest: $manifest"
}

_build_remote_command() {
  local rank=$1
  local log_file=$2
  local pid_file=$3
  local cmd="set -euo pipefail;"
  local mode=${CODE_SYNC_MODE:-verify}
  local item key value

  if [[ "$mode" == "reset" ]]; then
    cmd+=" if [[ ! -d $(_shell_quote "$LOCAL_REPO/.git") ]]; then"
    cmd+=" git clone -q $(_shell_quote "$REPO_URL") $(_shell_quote "$LOCAL_REPO");"
    cmd+=" fi;"
    cmd+=" git -C $(_shell_quote "$LOCAL_REPO") fetch -q origin $(_shell_quote "$CODE_REV");"
    cmd+=" git -C $(_shell_quote "$LOCAL_REPO") checkout -q --detach $(_shell_quote "$CODE_REV");"
    cmd+=" git -C $(_shell_quote "$LOCAL_REPO") reset -q --hard $(_shell_quote "$CODE_REV");"
  elif [[ "$mode" != "verify" ]]; then
    echo "ERROR: CODE_SYNC_MODE must be 'verify' or 'reset', got '$mode'" >&2
    return 2
  fi

  cmd+=" actual_rev=\$(git -C $(_shell_quote "$LOCAL_REPO") rev-parse HEAD);"
  cmd+=" if [[ \"\$actual_rev\" != $(_shell_quote "$CODE_REV") ]]; then"
  cmd+=" echo \"ERROR: code revision mismatch: expected $CODE_REV, got \$actual_rev\" >&2; exit 3; fi;"
  cmd+=" cd $(_shell_quote "$LOCAL_REPO/src");"
  if [[ -n ${VENV_DIR:-} ]]; then
    cmd+=" export PATH=$(_shell_quote "$VENV_DIR/bin"):\$PATH;"
  else
    cmd+=" source $(_shell_quote "$CONDA_SH");"
    cmd+=" conda activate $(_shell_quote "$CONDA_ENV");"
  fi

  for item in "${REMOTE_ENV[@]}"; do
    key=${item%%=*}
    value=${item#*=}
    if [[ ! "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
      echo "ERROR: invalid environment variable name '$key'" >&2
      return 2
    fi
    cmd+=" export $key=$(_shell_quote "$value");"
  done

  cmd+=" nohup torchrun"
  for item in "${TORCHRUN_COMMON_ARGS[@]}"; do
    _append_word cmd "$item"
  done
  _append_word cmd "--node_rank=$rank"
  for item in "${TRAIN_ARGS[@]}"; do
    _append_word cmd "$item"
  done
  cmd+=" > $(_shell_quote "$log_file") 2>&1 < /dev/null &"
  cmd+=" echo \$! > $(_shell_quote "$pid_file");"
  cmd+=" echo launched_rank=$rank pid=\$! revision=\$actual_rev;"
  printf '%s' "$cmd"
}

launch_multinode() {
  _require_scalar REPO_URL
  _require_scalar CODE_REV
  _require_scalar LOCAL_REPO
  if [[ -n ${VENV_DIR:-} ]]; then
    _require_scalar VENV_DIR
  else
    _require_scalar CONDA_SH
    _require_scalar CONDA_ENV
  fi
  _require_scalar LOGDIR
  _require_scalar OUT

  if ! declare -p NODES >/dev/null 2>&1 || ((${#NODES[@]} == 0)); then
    echo "ERROR: NODES array is empty" >&2
    return 2
  fi
  if ! declare -p TORCHRUN_COMMON_ARGS >/dev/null 2>&1; then
    echo "ERROR: TORCHRUN_COMMON_ARGS array is not defined" >&2
    return 2
  fi
  if ! declare -p TRAIN_ARGS >/dev/null 2>&1 || ((${#TRAIN_ARGS[@]} == 0)); then
    echo "ERROR: TRAIN_ARGS array is empty" >&2
    return 2
  fi
  if ! declare -p REMOTE_ENV >/dev/null 2>&1; then
    REMOTE_ENV=()
  fi

  mkdir -p "$LOGDIR" "$OUT" "$OUT/run_manifest"
  local timestamp
  timestamp=$(date -u +%Y%m%dT%H%M%SZ)
  _write_launch_manifest "$OUT/run_manifest" "$timestamp"

  local -a ssh_pids=()
  local rank node remote_cmd
  for rank in "${!NODES[@]}"; do
    node=${NODES[$rank]}
    remote_cmd=$(_build_remote_command \
      "$rank" "$LOGDIR/mn_node${rank}.log" "$LOGDIR/mn_node${rank}.pid")
    if [[ ${DRY_RUN:-0} == 1 ]]; then
      printf 'DRY-RUN node=%q rank=%q command=%q\n' "$node" "$rank" "$remote_cmd"
      continue
    fi
    echo "launching rank $rank on $node"
    ssh -o LogLevel=ERROR -o StrictHostKeyChecking=no "$node" "$remote_cmd" &
    ssh_pids+=("$!")
  done

  if [[ ${DRY_RUN:-0} == 1 ]]; then
    echo "dry-run complete; no remote process was started"
    return 0
  fi

  local failed=0 pid
  for pid in "${ssh_pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if ((failed)); then
    echo "ERROR: at least one remote launch failed; inspect launcher output" >&2
    return 1
  fi

  echo "launched ${#NODES[@]} nodes; logs: $LOGDIR/mn_node*.log"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "ERROR: do not run this generic launcher directly." >&2
  echo "Run a tracked script under recipes/ instead." >&2
  exit 2
fi
