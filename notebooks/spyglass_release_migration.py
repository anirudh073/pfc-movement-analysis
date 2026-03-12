#Author: GPT-5.3-Codex

"""Idempotent Spyglass release migration helper. For spyglass 0.5.5. Modified from release notes.

Usage from a notebook:
    from notebooks.spyglass_release_migration import run_spyglass_release_migration
    run_spyglass_release_migration()
"""

from __future__ import annotations

from typing import Iterable

import datajoint as dj


def _refresh_dependencies(connection: dj.Connection) -> None:
    connection.dependencies.load(force=True)


def _table_exists(connection: dj.Connection, full_name: str) -> bool:
    _refresh_dependencies(connection)
    return full_name in connection.dependencies.nodes


def _descendants_safe(connection: dj.Connection, full_name: str) -> list[str]:
    _refresh_dependencies(connection)
    if full_name not in connection.dependencies.nodes:
        return []
    return list(connection.dependencies.descendants(full_name))


def _drop_if_present(
    connection: dj.Connection,
    full_name: str,
    dry_run: bool = False,
) -> None:
    if not _table_exists(connection, full_name):
        print(f"SKIP: {full_name} does not exist (already dropped or never created).")
        return

    if dry_run:
        print(f"DRY RUN: would drop {full_name}")
        return

    dj.FreeTable(connection, full_name).drop()
    print(f"DROPPED: {full_name}")


def _try_call(label: str, func, stop_on_error: bool = False) -> None:
    try:
        func()
        print(f"OK: {label}")
    except Exception as exc:  # noqa: BLE001
        print(f"WARN: {label} -> {type(exc).__name__}: {exc}")
        if stop_on_error:
            raise


def _run_update_ids(
    label: str,
    update_ids_func,
    ignore_missing_analysis_files: bool,
) -> None:
    try:
        update_ids_func()
        print(f"OK: {label}")
    except FileNotFoundError as exc:
        print(
            "WARN: "
            f"{label} skipped due to missing analysis NWB file.\n"
            f"{exc}\n"
            "This indicates stale DB records pointing to files that are no longer on disk. "
            "Restore the missing file or remove stale entries before re-running update_ids."
        )
        if not ignore_missing_analysis_files:
            raise


def _run_v1_update_ids_per_row(ignore_missing_analysis_files: bool) -> None:
    """Run v1 update_ids with per-row missing-file handling.

    Spyglass v1's built-in update_ids aborts on first missing analysis NWB file.
    This helper reproduces the same update logic but continues past missing files.
    """
    import spyglass.spikesorting.v1.recording as v1_recording

    table = v1_recording.SpikeSortingRecording()
    elect_attr = "acquisition/ProcessedElectricalSeries/electrodes"
    needs_update = table & "electrodes_id is NULL or hash is NULL"

    updated_count = 0
    checksum_skipped_count = 0
    missing_paths: list[str] = []

    for key in v1_recording.tqdm(needs_update, desc="Updating v1 ids"):
        try:
            analysis_file_path = v1_recording.AnalysisNwbfile.get_abs_path(
                key["analysis_file_name"]
            )
        except dj.DataJointError:
            checksum_skipped_count += 1
            continue

        try:
            with v1_recording.H5File(analysis_file_path, "r") as h5_file:
                elect_id = h5_file[elect_attr].attrs["object_id"]
        except FileNotFoundError:
            missing_paths.append(analysis_file_path)
            if not ignore_missing_analysis_files:
                raise
            continue

        updated = dict(
            key,
            electrodes_id=elect_id,
            hash=v1_recording.NwbfileHasher(analysis_file_path).hash,
        )
        table.update1(updated)
        updated_count += 1

    print(
        "v1 update_ids summary: "
        f"updated={updated_count}, "
        f"missing_files_skipped={len(missing_paths)}, "
        f"checksum_skipped={checksum_skipped_count}"
    )
    if missing_paths:
        preview_n = 5
        print(
            "Missing analysis NWB files (first "
            f"{preview_n}):\n" + "\n".join(missing_paths[:preview_n])
        )


def _run_lfp_migration(dry_run: bool = False) -> None:
    from spyglass.lfp.lfp_imported import ImportedLFP
    from spyglass.lfp.lfp_merge import LFPOutput

    imported_lfp_count = len(ImportedLFP())
    merged_imported_count = len(LFPOutput.ImportedLFP())

    if imported_lfp_count or merged_imported_count:
        if dry_run:
            print(
                "DRY RUN: LFP migration is blocked because existing entries would be dropped.\n"
                f"ImportedLFP: {imported_lfp_count}\n"
                f"LFPOutput.ImportedLFP: {merged_imported_count}\n"
                "To proceed, either remove those entries first or run the migration with include_lfp=False."
            )
            return
        raise ValueError(
            "Existing entries found and would be dropped in update. "
            "Please delete entries first or request migration assistance.\n"
            f"ImportedLFP: {imported_lfp_count}\n"
            f"LFPOutput.ImportedLFP: {merged_imported_count}"
        )

    table = LFPOutput().ImportedLFP()
    connection = table.connection
    table_name = table.full_table_name

    if not _table_exists(connection, table_name):
        print(f"SKIP: {table_name} does not exist; LFP drop step not required.")
        return

    descendants = _descendants_safe(connection, table_name)
    if len(descendants) > 1:
        downstream = [name for name in descendants if name != table_name]
        raise ValueError(
            "Downstream tables exist and would be dropped in update.\n"
            "Drop these tables first:\n"
            + "\n".join(downstream)
        )

    imported_lfp_name = ImportedLFP().full_table_name
    if dry_run:
        print(f"DRY RUN: would drop_quick {table_name}")
        if _table_exists(connection, imported_lfp_name):
            print(f"DRY RUN: would drop {imported_lfp_name}")
        return

    table.drop_quick()
    print(f"DROPPED QUICK: {table_name}")

    if _table_exists(connection, imported_lfp_name):
        ImportedLFP().drop()
        print(f"DROPPED: {imported_lfp_name}")
    else:
        print(f"SKIP: {imported_lfp_name} does not exist (already dropped).")


def run_spyglass_release_migration(
    dry_run: bool = False,
    include_v0: bool = True,
    include_v1: bool = True,
    include_lfp: bool = True,
    stop_on_error: bool = False,
    ignore_missing_analysis_files: bool = True,
) -> None:
    """Run Spyglass release migration steps safely and repeatedly.

    Args:
        dry_run: Only report actions without changing DB tables.
        include_v0: Run v0 spikesorting alter/update_ids steps.
        include_v1: Run v1 spikesorting alter/update_ids steps.
        include_lfp: Run LFP migration drop step with safety checks.
        stop_on_error: Raise immediately on first step failure.
        ignore_missing_analysis_files: Continue when update_ids hits missing
            analysis NWB files referenced by stale DB records.
    """

    connection = dj.conn()
    print("Starting Spyglass release migration")
    print(
        "dry_run="
        f"{dry_run} include_v0={include_v0} include_v1={include_v1} "
        f"include_lfp={include_lfp} stop_on_error={stop_on_error} "
        f"ignore_missing_analysis_files={ignore_missing_analysis_files}"
    )

    # -- TrackGraph --
    def _trackgraph_alter() -> None:
        from spyglass.linearization.v1.main import TrackGraph

        if dry_run:
            print("DRY RUN: would run TrackGraph.alter()")
            return
        TrackGraph.alter()

    _try_call("TrackGraph.alter()", _trackgraph_alter, stop_on_error=stop_on_error)

    # -- Drop deprecated tables --
    deprecated_tables: Iterable[str] = (
        "`common_nwbfile`.`analysis_nwbfile_log`",
        "`common_session`.`session_group`",
    )
    for full_name in deprecated_tables:
        _try_call(
            f"drop deprecated table {full_name}",
            lambda n=full_name: _drop_if_present(connection, n, dry_run=dry_run),
            stop_on_error=stop_on_error,
        )

    # -- v0 recompute --
    if include_v0:
        def _v0_steps() -> None:
            import spyglass.spikesorting.v0.spikesorting_recording as v0_recording

            if dry_run:
                print("DRY RUN: would run v0 SpikeSortingRecording.alter()")
                print("DRY RUN: would run v0 SpikeSortingRecording.update_ids()")
                return
            # Use module context so DataJoint can resolve FK symbols in definitions.
            v0_recording.SpikeSortingRecording().alter(context=v0_recording.__dict__)
            _run_update_ids(
                "v0 SpikeSortingRecording.update_ids()",
                v0_recording.SpikeSortingRecording().update_ids,
                ignore_missing_analysis_files=ignore_missing_analysis_files,
            )

        _try_call(
            "v0 SpikeSortingRecording alter/update_ids",
            _v0_steps,
            stop_on_error=stop_on_error,
        )
    else:
        print("SKIP: v0 migration disabled by include_v0=False")

    # -- v1 recompute --
    if include_v1:
        def _v1_steps() -> None:
            import spyglass.spikesorting.v1.recording as v1_recording

            if dry_run:
                print("DRY RUN: would run v1 SpikeSortingRecording.alter()")
                print("DRY RUN: would run v1 SpikeSortingRecording.update_ids()")
                return
            # Use module context so DataJoint can resolve FK symbols in definitions.
            v1_recording.SpikeSortingRecording().alter(context=v1_recording.__dict__)
            _run_v1_update_ids_per_row(
                ignore_missing_analysis_files=ignore_missing_analysis_files
            )

        _try_call(
            "v1 SpikeSortingRecording alter/update_ids",
            _v1_steps,
            stop_on_error=stop_on_error,
        )
    else:
        print("SKIP: v1 migration disabled by include_v1=False")

    # -- LFP pipeline --
    if include_lfp:
        _try_call(
            "LFP migration drop step",
            lambda: _run_lfp_migration(dry_run=dry_run),
            stop_on_error=stop_on_error,
        )
    else:
        print("SKIP: LFP migration disabled by include_lfp=False")

    print("Migration run complete")
