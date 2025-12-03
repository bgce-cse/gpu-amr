#!/usr/bin/env python3
"""
json_to_vtk.py

Reads JSON files produced by export_patch_dofs_json and converts them to VTK format.
The JSON format includes metadata, points, connectivity, and point_data (DOF components).

Usage:
  python3 tools/json_to_vtk.py vtk_from_json dg_timestep

This writes timestep-based VTK files and a PVD collection file.
"""
import json
import sys
import os

def write_legacy_vtk(filename, points, connectivity, point_data_arrays, array_names):
    """
    Write legacy VTK file with unstructured grid.
    
    Args:
        filename: Output VTK file path
        points: List of [x, y, z] coordinates
        connectivity: List of cell connectivities, each is list of point indices
        point_data_arrays: List of arrays, one per DOF component
        array_names: Names for each array
    """
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
        
        # Write cell types (all VTK_POLYGON for now, or use VTK_QUAD/VTK_HEXAHEDRON)
        f.write(f'CELL_TYPES {len(connectivity)}\n')
        for cell in connectivity:
            # Determine cell type based on number of points
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

def main():
    if len(sys.argv) < 3:
        print('Usage: json_to_vtk.py <json_dir> <output_prefix>')
        return 1

    json_dir = sys.argv[1]
    prefix   = sys.argv[2]

    os.makedirs('vtk_from_json', exist_ok=True)

    # Check if json_dir is a directory
    if not os.path.isdir(json_dir):
        print(f'Error: {json_dir} is not a directory')
        return 1

    # Find all JSON files matching pattern dofs_t{t}_p{idx}.json
    import re
    files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    
    # Group by timestep
    groups = {}
    pat = re.compile(r'dofs_t(\d+)_p(\d+)\.json')
    
    for fname in files:
        m = pat.match(fname)
        if not m:
            continue
        t = int(m.group(1))
        p = int(m.group(2))
        groups.setdefault(t, []).append((p, os.path.join(json_dir, fname)))
    
    if not groups:
        print(f'No matching JSON files found in {json_dir}')
        return 1

    timesteps = sorted(groups.keys())
    vtk_files = []
    pvd_times = []
    
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
                            # Pad with zeros if missing
                            all_point_data[name].extend([0.0] * len(points))
            
            except Exception as e:
                print(f'Warning: Error processing {json_file}: {e}')
                continue
        
        if not all_points:
            print(f'Warning: No points found for timestep {t}')
            continue
        
        # Convert point_data dict to list of arrays in sorted order
        point_data_arrays = []
        if all_point_data:
            for name in sorted(array_names):
                point_data_arrays.append(all_point_data[name])
        
        # Write VTK file
        vtk_filename = os.path.join('vtk_from_json', f'{prefix}_t{t}.vtk')
        write_legacy_vtk(vtk_filename, all_points, all_connectivity, 
                        point_data_arrays, sorted(array_names))
        
        vtk_files.append(os.path.basename(vtk_filename))
        pvd_times.append(float(t))
        print(f'Wrote {vtk_filename}')
    
    # Write PVD collection file
    pvd_name = os.path.join('vtk_from_json', f'{prefix}.pvd')
    with open(pvd_name, 'w') as pvdf:
        pvdf.write('<?xml version="1.0"?>\n')
        pvdf.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        pvdf.write('  <Collection>\n')
        for fname, time_val in zip(vtk_files, pvd_times):
            pvdf.write(f'    <DataSet timestep="{time_val}" group="" part="0" file="{fname}"/>\n')
        pvdf.write('  </Collection>\n')
        pvdf.write('</VTKFile>\n')
    
    print(f'\nWrote {len(vtk_files)} timestep VTK files')
    print(f'Wrote PVD collection: {pvd_name}')
    return 0

if __name__ == '__main__':
    sys.exit(main())

