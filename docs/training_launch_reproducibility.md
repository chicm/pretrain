# Reproducible multi-node training launches

Training jobs use a two-layer launch layout:

- `recipes/*.sh` is the experiment record. A recipe intentionally contains the real topology, filesystem paths, source revision, environment, and complete `torchrun`/`train.py` arguments.
- `scripts/launch_multinode.sh` is generic orchestration code. It contains no cluster-specific paths or experiment hyperparameters.

The canonical 1T recipe is:

```text
recipes/chimera_8b_1t.sh
```

Its header records the exact detached command used on the launcher node. Do not maintain a second untracked launch script on shared storage; if an operational copy is required, copy it from the pinned Git revision and retain the generated manifest.

## Launch contract

A recipe defines:

- `REPO_URL`, `CODE_REV`, `CODE_SYNC_MODE`, and `LOCAL_REPO`;
- the `NODES` array;
- conda/runtime paths;
- data, output, log, and TensorBoard paths;
- `REMOTE_ENV`;
- `TORCHRUN_COMMON_ARGS`;
- the complete `TRAIN_ARGS` array.

It then sources the generic launcher and calls:

```bash
launch_multinode
```

The generic launcher shell-quotes every argument before SSH execution, verifies every worker's Git revision, and refuses to launch a mixed-revision job.

## Source deployment

The recommended production flow separates deployment from launch:

1. Resolve an immutable Git commit.
2. Deploy that commit to the configured local checkout on every node using an account with the required filesystem permissions.
3. Keep `CODE_SYNC_MODE=verify` in the recipe.
4. Launch under the account that owns inter-node SSH trust.

`CODE_SYNC_MODE=reset` is available only when the launcher account is allowed to clone/fetch/reset the checkout on every node. A failed reset or revision mismatch is fatal; the launcher never silently continues with stale code.

## Runtime manifest

Each launch writes the following next to its checkpoint directory:

```text
run_manifest/
  launch-<UTC timestamp>.txt
  launch-<UTC timestamp>.cmd
  launch-<UTC timestamp>.recipe.sh
  latest.txt
```

The manifest records the recipe, source revision, nodes, environment allow-list, `torchrun` arguments, and training arguments. It must not contain access tokens or arbitrary inherited environment variables.

The copied recipe and command file preserve what was actually requested even if the branch later moves.

## Resume

A resume is a new launch and therefore creates a new timestamped manifest. The recipe should retain `--resume latest` when the intended behavior is to restore the newest complete checkpoint. Before resuming:

1. verify that all old ranks have exited;
2. deploy/verify the recipe's source revision;
3. launch the tracked recipe;
4. confirm the restored step in rank-0 logs;
5. verify at least one subsequent checkpoint round-trip.

## Deprecated entry points

The former root and `src/` multi-node shell entry points are compatibility shims and intentionally fail with a pointer to `recipes/`. This prevents stale launchers from silently diverging from the command used for production training.
