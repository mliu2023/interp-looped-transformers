# GPU smoke submission

Recovery job **7496249** completed successfully with exit code 0:0. It installed
torch 2.8.0+cu128 / torchvision 0.23.0+cu128, verified the staged inputs and created
snapshot b7f0f4ae8ff760f8d79f6bce6a702cf14f731e34f5b14c98429b53103fd779ec.
Its full CPU suite ran 115 tests without failures, with one source-path skip.
That source-path test now honors REUSE_LOOPVL_DIR and passed against the actual
pinned processor. All **34 focused scheduler/alignment tests passed with no skips**.

The editable submitter now permits a repaired validation snapshot only after a
failed, known, terminal validation attempt of the same study and shard count.
Successful, active, unknown and research-shard hash mismatches remain rejected.
The frozen experiment snapshot is unchanged; the updated orchestration wrapper
hash is saved in the submission evidence.

Slurm accepted GPU numerical smoke **job 7501493**. It requests one A6000 GPU,
8 CPUs, 32 GiB host RAM and one hour. Fresh accounting records 0.0047222222
GPU-hours consumed and 1 GPU-hour reserved, leaving 28.9952777778 GPU-hours of
the 30-hour cap. No scientific endpoints have been measured.

Evidence: results/cluster-execution-20261016/gpu-smoke-20261017T045104Z/.
At the next requested continuation, inspect the study base's
runs/logs/7501493.out, runs/logs/7501493.err,
runs/preflight/7501493_single.json and runs/loopvl-transfer-v1/parity_report.json.
Require successful numerical parity before the 32-example pilot.
Use the editable source/scripts/slurm/submit.py wrapper to orchestrate this
snapshot; its frozen older submitter retains the failed-snapshot retry bug.
Site policy prohibits automatic polling.

