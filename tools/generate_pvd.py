#!/usr/bin/env python3
"""
Generate a PVD (ParaView Data) collection file from existing VTK files.
Scans the vtk_output directory for VTK files matching the pattern and creates a time-series collection.
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple

def extract_timestep(filename: str) -> int:
    """Extract timestep number from filename."""
    # Try pattern with Order info: dg_tree_advanced_timestep_Order2_N.vtk
    match = re.search(r'_(\d+)\.vtk$', filename)
    if match:
        return int(match.group(1))
    return -1

def get_vtk_files(vtk_dir: Path) -> List[Tuple[str, int, float]]:
    """
    Get all VTK files from directory, sorted by timestep.
    Returns list of (filename, timestep, time) tuples.
    """
    files = []
    
    # Find all .vtk files (handles both patterns with and without Order info)
    for vtk_file in sorted(vtk_dir.glob('dg_tree_advanced_timestep*.vtk')):
        # Skip the .pvd file if it exists
        if vtk_file.name.endswith('.pvd'):
            continue
        timestep = extract_timestep(vtk_file.name)
        if timestep >= 0:
            time = int(timestep)
            files.append((vtk_file.name, timestep, time))
    
    return sorted(files, key=lambda x: x[1])

def write_pvd(pvd_path: Path, vtk_files: List[Tuple[str, int, float]]) -> None:
    """Write PVD collection file with relative paths."""
    vtk_dir = pvd_path.parent
    
    with open(pvd_path, 'w') as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1">\n')
        f.write('  <Collection>\n')
        
        for filename, timestep, time in vtk_files:
            # Use relative paths - ParaView prefers these
            f.write(f'    <DataSet timestep="{time:.1f}" file="{filename}"/>\n')
        
        f.write('  </Collection>\n')
        f.write('</VTKFile>\n')

def main():
    # Paths
    base_dir = Path(__file__).parent.parent
    vtk_dir = base_dir / "build" / "vtk_output"
    pvd_file = vtk_dir / "dg_tree_advanced_timestep.pvd"
    
    # Check if vtk_output directory exists
    if not vtk_dir.exists():
        print(f"Error: {vtk_dir} does not exist")
        sys.exit(1)
    
    # Get VTK files
    vtk_files = get_vtk_files(vtk_dir)
    
    if not vtk_files:
        print(f"Error: No VTK files found in {vtk_dir}")
        sys.exit(1)
    
    print(f"Found {len(vtk_files)} VTK files")
    print(f"Timestep range: {vtk_files[0][1]} to {vtk_files[-1][1]}")
    print(f"Time range: {vtk_files[0][2]} to {vtk_files[-1][2]}")
    
    # Write PVD file
    write_pvd(pvd_file, vtk_files)
    print(f"\n✓ PVD file created: {pvd_file}")
    print(f"✓ Total timesteps: {len(vtk_files)}")

if __name__ == "__main__":
    main()
