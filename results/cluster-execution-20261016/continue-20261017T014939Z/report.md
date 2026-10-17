# Cluster continuation

Networking is restored. The single queue/accounting check found no active jobs and no earlier reuseinterp-env submission.

Slurm accepted CPU environment recovery job **7496249**. It requests one CPU, 16 GiB host RAM and one hour, with no GPU. Acceptance is recorded in recovery_submission.json; completion has not been checked.

The job creates the isolated CUDA 12.8 environment, runs the full CPU suite, verifies staged inputs and freezes a new snapshot. No scientific endpoint has been measured. The next gate is the scheduled GPU numerical smoke, after recovery actually succeeds.

The site policy prohibits automatic polling; inspect results at the next user-requested continuation.
