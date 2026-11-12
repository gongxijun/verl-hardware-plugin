# Enflame GCU feature support

This page records what the plugin delivers for Enflame GCU, and how far each
item has been validated. Enflame GCU remains **`Example (requires vendor
support)`** in the repository status table until the evidence in
[Pending evidence](#pending-evidence) exists; nothing here promotes the status.

Tiers used below:

| Tier | Meaning |
|------|---------|
| ✅ Feasible | Implemented in the plugin, using only APIs present in verl `main`. |
| ⚠️ Partially feasible | Implemented, but depends on a vendor component or still needs real-hardware validation. |
| ❌ Not feasible | Blocked by an upstream removal or a missing vendor capability; not attempted. |
| Not provided (out of scope) | Deliberately excluded from this plugin; not a statement that the hardware cannot do it. |

## Summary

| Capability | Tier | Where |
|---|---|---|
| Platform registration (`device_name="gcu"`, `vendor_name="enflame"`) | ✅ | `verl_hardware_plugin/accelerators/enflame/platform_enflame.py` |
| FSDP training engine | ✅ | `verl_hardware_plugin/accelerators/enflame/engines/fsdp_enflame.py` |
| Megatron training engine | ⚠️ | `verl_hardware_plugin/accelerators/enflame/engines/megatron_enflame.py`; needs Megatron-LM-FL |
| Checkpoint engine (disaggregated weight transfer) | ⚠️ | `verl_hardware_plugin/accelerators/enflame/engines/eccl_checkpoint_engine.py`; ECCL/FlagCX over `ray.util.collective` |
| Colocated weight update | ⚠️ | verl's device-side IPC handle path (`reduce_tensor` + `rebuild_ipc`); layout unverified on GCU |
| Python multiprocessing SHM / IPC handle weight transfer | ⚠️ | `PlatformENFLAME.is_ipc_supported()`; CUDA-style handle layout unverified |

## Training engines

### FSDP — ✅ Feasible

`fsdp_enflame` registers the FSDP strategy for `(device="gcu", vendor="enflame")`
and is the path used by the acceptance baseline. It still requires the vendor
training kernels (TransformerEngine-FL) installed in every Ray worker, as
described in [install_guidance.md](./install_guidance.md).

### Megatron — ⚠️ Partially feasible

`megatron_enflame` is registered, but Megatron-LM-FL must be installed and on
`PYTHONPATH` for the driver and every Ray worker. On a node without it the
registration is skipped with a debug log and the FSDP path is unaffected.

## Checkpoint engine — ⚠️ Partially feasible

`ECCLCheckpointEngine` implements the same protocol as the in-tree NCCL engines:
per-bucket tensor metadata travels over zeromq, while the bucket bytes
are broadcast with `ray.util.collective` and two alternating device buffers.

Two names are involved and must not be confused:

- **Registry/config name** (`eccl`, `nccl`): the key verl looks up in
  `CheckpointEngineRegistry`. This engine is registered under **both** names.
- **Collective backend name** (`eccl`, `flagcx`): the backend argument passed to
  `ray.util.collective`. It follows
  `PlatformENFLAME.communication_backend_name()`, so `USE_FLAGCX=1` selects
  `flagcx` and the default selects `eccl`.

`nccl` is a compatibility alias, not the default: verl's `rollout.yaml` defaults
to `backend: naive`, and only scripts that opt into the v1 async path (for
example `tests/special_e2e/run_v1_separate_async.sh`) set
`checkpoint_engine.backend=nccl`. On GCU verl's ECCL engine claims that alias, 
so such scripts run unchanged. verl's own NCCL engine is currently unusable on
GCU (it needs cupy and assumes `"cuda"`). If a future build makes it importable
there, this alias will override
it, so remove the alias at that point.

- The module refuses to import unless `torch.gcu.is_available()` is true. This
  matters because the plugin is loaded on every node: without the guard a CUDA
  host that merely has `torch_gcu` installed would lose its working NCCL engine.
- `build_topology()` keeps only actor rank 0 in the collective group
  (`rank=[0, -1, ...]`, `world_size=rollout_world_size + 1`), matching the
  existing implementations.

What is still unverified: a multi-node, disaggregated or async run on real GCUs.
The logic is a direct port, but the ECCL/FlagCX group creation, bandwidth and
zeromq behavior have not been exercised here.

### Configuration

Use the canonical registry name for a disaggregated or async rollout:

```bash
python -m verl.trainer.main_ppo \
  ... \
  actor_rollout_ref.rollout.checkpoint_engine.backend=eccl
```

- `backend=eccl` is the documented name. `backend=nccl` is a compatibility alias
  and resolves to the same engine.
- The checkpoint engine is only instantiated for disaggregated/async rollout;
  colocated rollout does not use it.
- `actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes`
  controls the bucket size (default `2048`).
- `USE_FLAGCX=1` keeps the registry name unchanged but switches the collective
  backend to FlagCX through the platform. Because Ray workers do not inherit
  shell exports, `USE_FLAGCX` (and `FLAGCX_PATH` when it is set) must also be
  placed in `ray_init.runtime_env.env_vars`; otherwise workers resolve the
  platform backend as `eccl` while the driver uses FlagCX.

## Weight transfer and IPC (partially feasible)

`PlatformENFLAME.is_ipc_supported()` returns `True`, meaning "device-side IPC is
available". verl converts that into `use_shm = not is_support_ipc() = False`, so
it skips the Python multiprocessing shared-memory fallback (which torch_gcu does
not support). It does **not** mean "no IPC": the two rollout layouts then differ.

- **Colocated actor/rollout**: `vllm_rollout.py` takes the `use_shm=False` path,
  and `BucketedWeightSender` synchronizes weights through device-side IPC
  handles using `reduce_tensor` + `rebuild_ipc`. This is still IPC, just not
  Python SHM. ⚠️
- **Disaggregated/async actor/rollout**: the ECCL checkpoint engine broadcasts
  on-device buffers; no IPC handle is involved. ⚠️ See above.

Known risk: `rebuild_ipc()` assumes a CUDA `PrivateUse1` handle layout —
`bucketed_weight_transfer.py` writes `list_args[6] = device_id`. Whether
torch_gcu's handle matches that layout must be validated on real hardware. Until
then, treat the colocated IPC-handle path as unverified. There is currently no
validated fallback for colocated rollout: confirm `rebuild_ipc` on real GCUs
first, otherwise switch to disaggregated rollout with the ECCL checkpoint engine.

## Upstream API dependencies

The checkpoint engine imports symbols that are internal to verl rather than a
stable public API. Before upgrading verl, verify they still exist with the same
semantics:

- `verl.checkpoint_engine.base`: `CheckpointEngine`, `CheckpointEngineRegistry`,
  `TensorMeta`, `split_weight_chunks`, `merge_weight_chunks`
- `verl.utils.net_utils`: `get_free_port`, `is_valid_ipv6_address`
- `verl.plugin.platform.get_platform` and
  `PlatformBase.communication_backend_name()`
- `ray.util.collective`: `init_collective_group`, `is_group_initialized`,
  `broadcast`, `barrier`, `destroy_collective_group`

If any of these change, update the engine and its CPU tests together.
