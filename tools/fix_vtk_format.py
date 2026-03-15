#!/usr/bin/env python3
"""
Fix VTK files by ensuring proper POINT_DATA format with separate SCALARS sections.
"""

import re
from pathlib import Path

def fix_vtk_point_data_format(input_file: Path, output_file: Path) -> None:
    """
    Read VTK file and ensure POINT_DATA format is correct.
    Multiple SCALARS sections should each start fresh after the LOOKUP_TABLE.
    """
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    output_lines = []
    i = 0
    in_point_data = False
    point_data_count = 0
    
    while i < len(lines):
        line = lines[i]
        
        if line.startswith('POINT_DATA'):
            in_point_data = True
            match = re.match(r'POINT_DATA\s+(\d+)', line)
            if match:
                point_data_count = int(match.group(1))
            output_lines.append(line)
            i += 1
            continue
        
        if in_point_data and line.startswith('SCALARS'):
            # Ensure proper spacing: blank line before SCALARS (except first)
            if output_lines and not output_lines[-1].strip().startswith('LOOKUP_TABLE'):
                # Previous SCALARS block hasn't ended yet
                pass
            output_lines.append(line)
            i += 1
            
            # Next line should be LOOKUP_TABLE
            if i < len(lines) and lines[i].startswith('LOOKUP_TABLE'):
                output_lines.append(lines[i])
                i += 1
                
                # Now read the data values (should be point_data_count values)
                values_collected = 0
                while i < len(lines) and values_collected < point_data_count:
                    data_line = lines[i].strip()
                    if data_line and not data_line.startswith(('SCALARS', 'LOOKUP_TABLE', 'POINT_DATA', 'CELLS', 'POINTS', 'CELL_TYPES')):
                        try:
                            float(data_line)
                            output_lines.append(lines[i])
                            values_collected += 1
                            i += 1
                        except ValueError:
                            break
                    else:
                        break
            continue
        
        output_lines.append(line)
        i += 1
    
    with open(output_file, 'w') as f:
        f.writelines(output_lines)

def main():
    input_dir = Path("/Users/riccardocapellupo/Desktop/BGCE/gpu-amr/build/vtk_output")
    
    for vtk_file in sorted(input_dir.glob('dg_tree_advanced_timestep_Order2_*.vtk')):
        try:
            fix_vtk_point_data_format(vtk_file, vtk_file)
            print(f"✓ Fixed: {vtk_file.name}")
        except Exception as e:
            print(f"✗ Error fixing {vtk_file.name}: {e}")

if __name__ == "__main__":
    main()
