# Scheduler retry

Retried the queue and accounting checks at the user's request. `squeue --me` timed out after 8 seconds; `sacct` could not connect to slurm01:6819.

On login010, creation of both IPv4 and IPv6 sockets raises PermissionError errno 1 (Operation not permitted). This session's network restrictions prevent scheduler access. No new submission was attempted; the previous CPU submission remains unconfirmed.

Session permissions must allow Slurm network access before another queue reconciliation and safe submission. No model work or additional GPU usage occurred in this retry.
