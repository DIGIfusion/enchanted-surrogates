# Packers

A packer archives the result of a finished run directory into one storage file, so that
the run directory itself can be cleaned up. The supervisor calls `pack_run_dir(run_dir,
results)` on the packer of the current runner for every run it sees finish. Packers are
configured under `packers:` and referenced by name from `run_order`.

```yaml
packers:
  packer1:
    type: ascii_bin_to_hdf5_packer

supervisor:
  run_order:
    - executor: ...
      sampler: ...
      runner: ...
      packer: packer1
```

## Packing the same run directory again

**A packer is responsible for handling a run directory that is already packed.** The
supervisor does not keep track of what was packed. `pack_run_dir` can be called again for
the same run directory when:

- a crashed or interrupted ES run is resumed (`run_mode: resume`): the runs that finished
  before the interruption are seen as finished again and are handed to the packer again;
- a run directory is deliberately rerun.

In both cases `pack_run_dir` must **overwrite** what is packed for that run directory and
never add a duplicate. Packing the same run directory twice leaves the same packed state as
packing it once. A packer that cannot do this makes a resumed run produce duplicate or
stale data.

Because the supervisor deletes the files it does not keep (`save_files`) after packing, a
run directory that is packed again after a resume may contain fewer files than when it was
first packed. The packer packs what is in the directory at that time.

## Available packers

- `ascii_bin_to_hdf5_packer`: stores every file of the run directory in an HDF5 file, under
  `runs/<run directory name>/`, as text or bytes. An existing group with the same name is
  deleted and replaced. Note that HDF5 does not give the space of a deleted group back, so
  the file grows when many runs are packed again.
