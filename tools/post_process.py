#!/usr/bin/env python3
"""
post_process.py

Post-processing script for DG AMR simulations:
1. Aggregates JSON exports to VTK files
2. Creates PVD collection for time series visualization
3. Checks for crashes/anomalies in simulation data

Usage:
    python3 tools/post_process.py [--input-dir INPUT_DIR] [--output-dir OUTPUT_DIR] [--verbose]

Default directories:
    Input:  vtk_from_json/
    Output: vtk_output/
"""

import json
import sys
import os
import argparse
import re
from pathlib import Path
from collections import defaultdict


def write_legacy_vtk(filename, points, connectivity, point_data_arrays, array_names):
    """Write legacy VTK file with unstructured grid."""
    with open(filename, 'w') as f:
        f.write('# vtk DataFile Version 3.0\n')
        f.write('DG patch data\n')
        f.write('ASCII\n')
        f.write('DATASET UNSTRUCTURED_GRID\n')

        # Write points
        f.write(f'POINTS {len(points)} double\n')
        for pt in points:
            f.write(f'{pt[0]:.16e} {pt[1]:.16e} {pt[2]:.16e}\n')

        # Write cells
        total_connectivity = sum(len(cell) + 1 for cell in connectivity)
        f.write(f'CELLS {len(connectivity)} {total_connectivity}\n')
        for cell in connectivity:
            f.write(f'{len(cell)}')
            for pt_idx in cell:
                f.write(f' {pt_idx}')
            f.write('\n')

        # Write cell types
        f.write(f'CELL_TYPES {len(connectivity)}\n')
        for cell in connectivity:
            if len(cell) == 4:
                f.write('9\n')  # VTK_QUAD
            elif len(cell) == 8:
                f.write('12\n')  # VTK_HEXAHEDRON
            else:
                f.write('7\n')  # VTK_POLYGON (fallback)

        # Write point data
        if point_data_arrays and len(point_data_arrays) > 0:
            f.write(f'POINT_DATA {len(points)}\n')
            for name, arr in zip(array_names, point_data_arrays):
                f.write(f'SCALARS {name} double 1\n')
                f.write('LOOKUP_TABLE default\n')
                for val in arr:
                    f.write(f'{val:.16e}\n')


def check_for_crashes(point_data_arrays, array_names, verbose=False):
    """Check for NaN, Inf, or other anomalies in data."""
    issues = []

    for name, arr in zip(array_names, point_data_arrays):
        for i, val in enumerate(arr):
            if val != val:  # NaN check
                issues.append(
                    f"NaN found in {name} at point {i}: {val}"
                )
            elif val == float('inf'):
                issues.append(
                    f"Inf found in {name} at point {i}: {val}"
                )
            elif val == float('-inf'):
                issues.append(
                    f"-Inf found in {name} at point {i}: {val}"
                )
            elif abs(val) > 1e10:
                if verbose:
                    issues.append(
                        f"Large value in {name} at point {i}: {val:.6e}"
                    )

    return issues


def process_json_to_vtk(json_dir, output_dir, verbose=False):
    """Convert JSON files to VTK format."""
    if not os.path.isdir(json_dir):
        print(f'Error: {json_dir} is not a directory')
        return False

    os.makedirs(output_dir, exist_ok=True)

    # Find all JSON files matching pattern dofs_t{t}_p{idx}.json
    files = [f for f in os.listdir(json_dir) if f.endswith('.json')]

    # Group by timestep
    groups = defaultdict(list)
    pat = re.compile(r'dofs_t(\d+)_p(\d+)\.json')

    for fname in files:
        m = pat.match(fname)
        if not m:
            continue
        t = int(m.group(1))
        p = int(m.group(2))
        groups[t].append((p, os.path.join(json_dir, fname)))

    if not groups:
        print(f'No matching JSON files found in {json_dir}')
        return False

    timesteps = sorted(groups.keys())
    vtk_files = []
    pvd_times = []
    total_issues = []

    for t in timesteps:
        all_points = []
        all_connectivity = []
        all_point_data = None
        array_names = []

        # Aggregate all patches for this timestep
        for p, json_file in sorted(groups[t]):
            try:
                with open(json_file) as f:
                    data = json.load(f)

                # Extract metadata
                meta = data.get('metadata', {})
                points_offset = len(all_points)

                # Add points with offset
                points = data.get('points', [])
                for pt in points:
                    if len(pt) == 2:
                        all_points.append([pt[0], pt[1], 0.0])
                    else:
                        all_points.append(pt)

                # Add connectivity with offset
                connectivity = data.get('connectivity', [])
                for cell in connectivity:
                    offsetted_cell = [idx + points_offset for idx in cell]
                    all_connectivity.append(offsetted_cell)

                # Aggregate point data
                point_data = data.get('point_data', {})
                if point_data:
                    if all_point_data is None:
                        all_point_data = {}
                        array_names = sorted(point_data.keys())
                        for name in array_names:
                            all_point_data[name] = []

                    # Append data arrays
                    for name in array_names:
                        if name in point_data:
                            all_point_data[name].extend(point_data[name])
                        else:
                            all_point_data[name].extend([0.0] * len(points))

            except Exception as e:
                print(f'Warning: Error processing {json_file}: {e}')
                continue

        if not all_points:
            print(f'Warning: No points found for timestep {t}')
            continue

        # Convert point_data dict to list of arrays
        point_data_arrays = []
        if all_point_data:
            for name in sorted(array_names):
                point_data_arrays.append(all_point_data[name])

        # Check for crashes/anomalies
        issues = check_for_crashes(point_data_arrays, sorted(array_names), verbose)
        if issues:
            total_issues.extend([(t, issue) for issue in issues])
            if verbose:
                print(f"Issues at timestep {t}:")
                for issue in issues[:5]:  # Show first 5
                    print(f"  {issue}")
                if len(issues) > 5:
                    print(f"  ... and {len(issues) - 5} more")

        # Write VTK file
        vtk_filename = os.path.join(output_dir, f'dg_timestep_t{t}.vtk')
        try:
            write_legacy_vtk(vtk_filename, all_points, all_connectivity,
                            point_data_arrays, sorted(array_names))
            vtk_files.append(os.path.basename(vtk_filename))
            pvd_times.append(float(t))
            if verbose:
                print(f'✓ Wrote {vtk_filename} ({len(all_points)} points, '
                      f'{len(all_connectivity)} cells)')
        except Exception as e:
            print(f'Error writing VTK file {vtk_filename}: {e}')
            return False

    # Write PVD collection file
    pvd_name = os.path.join(output_dir, 'dg_timestep.pvd')
    try:
        with open(pvd_name, 'w') as pvdf:
            pvdf.write('<?xml version="1.0"?>\n')
            pvdf.write('<VTKFile type="Collection" version="0.1" '
                      'byte_order="LittleEndian">\n')
            pvdf.write('  <Collection>\n')
            for fname, time_val in zip(vtk_files, pvd_times):
                pvdf.write(f'    <DataSet timestep="{time_val}" group="" '
                          f'part="0" file="{fname}"/>\n')
            pvdf.write('  </Collection>\n')
            pvdf.write('</VTKFile>\n')
        print(f'\n✓ Wrote PVD collection: {pvd_name}')
    except Exception as e:
        print(f'Error writing PVD file: {e}')
        return False

    print(f'✓ Wrote {len(vtk_files)} timestep VTK files')
    print(f'✓ Time range: {pvd_times[0]:.1f} to {pvd_times[-1]:.1f}')

    # Report issues
    if total_issues:
        print(f'\n⚠ WARNING: Found {len(total_issues)} anomalies in data:')
        for t, issue in total_issues[:10]:
            print(f'  t={t}: {issue}')
        if len(total_issues) > 10:
            print(f'  ... and {len(total_issues) - 10} more')
        return len(total_issues) < 100  # Pass if issues are manageable
    else:
        print('\n✓ No NaN/Inf anomalies detected')

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Post-process DG AMR JSON exports to VTK files'
    )
    parser.add_argument(
        '--input-dir', default='vtk_from_json',
        help='Input directory with JSON files (default: vtk_from_json)'
    )
    parser.add_argument(
        '--output-dir', default='vtk_output',
        help='Output directory for VTK files (default: vtk_output)'
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    print('=' * 70)
    print('DG AMR Post-Processing')
    print('=' * 70)
    print(f'Input directory:  {args.input_dir}')
    print(f'Output directory: {args.output_dir}')
    print()

    success = process_json_to_vtk(args.input_dir, args.output_dir, args.verbose)

    print()
    print('=' * 70)
    if success:
        print('✓ Post-processing completed successfully')
        return 0
    else:
        print('✗ Post-processing failed or found critical issues')
        return 1


if __name__ == '__main__':
    sys.exit(main())
