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

A packer never adds a duplicate. What it does instead is set by `on_existing`:

```yaml
packers:
  packer1:
    type: ascii_bin_to_hdf5_packer
    on_existing: skip        # default; or: overwrite
```

- `skip` (default): what is already packed is kept and the call does nothing. This is the
  safe choice for resuming. The supervisor deletes the files it does not keep
  (`save_files`) after packing, so a run directory that is packed again after a resume is
  often almost empty, and overwriting would replace the complete packed copy with that.
- `overwrite`: what is packed for the run directory is replaced by its current content. Use
  it when run directories are rerun and the packed data must be refreshed. The packers
  then also overwrite after a resume, so keep the files you want to preserve in
  `save_files_list`.

Packing the same run directory twice leaves the same packed state as packing it once in both
modes. A custom packer must do the same, or a resumed run produces duplicate or stale data.

## Available packers

- `ascii_bin_to_hdf5_packer`: stores every file of the run directory in an HDF5 file, under
  `runs/<run directory name>/`, as text or bytes. With `on_existing: overwrite` an existing
  group with the same name is deleted and replaced; note that HDF5 does not give the space
  of a deleted group back, so the file grows when many runs are packed again.
- `csv_packer`: appends the parsed results of a run (the `results` dict, plus a `run_dir`
  column) as one row to a csv file. A run directory that is already in the file keeps its
  row (`skip`) or has the row replaced in place (`overwrite`); there is never a second row.
- `multi_packer`: runs several packers in turn on every run directory, so one packer
  failing does not stop the others.
