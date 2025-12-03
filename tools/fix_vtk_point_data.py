#!/usr/bin/env python3
"""
Create a minimal valid VTK file from the advanced printer's output.
Fixes ParaView crashes by correcting POINT_DATA array sizes.
"""

import re
from pathlib import Path

def fix_vtk_file(input_file: Path, output_file: Path) -> bool:
    """
    Read VTK file, extract POINTS count, and ensure POINT_DATA matches.
    Returns True if file was fixed, False otherwise.
    """
    try:
        with open(input_file, 'r') as f:
            content = f.read()
        
        # Extract POINTS count
        points_match = re.search(r'POINTS\s+(\d+)\s+double', content)
        if not points_match:
            print(f"Error: Could not find POINTS declaration in {input_file}")
            return False
        
        points_count = int(points_match.group(1))
        
        # Find POINT_DATA section
        pointdata_match = re.search(r'POINT_DATA\s+(\d+)', content)
        if not pointdata_match:
            print(f"Error: Could not find POINT_DATA declaration in {input_file}")
            return False
        
        pointdata_count = int(pointdata_match.group(1))
        
        if points_count == pointdata_count:
            print(f"✓ {input_file.name}: Points ({points_count}) match POINT_DATA ({pointdata_count})")
            return True
        
        print(f"✗ {input_file.name}: Points mismatch - POINTS={points_count}, POINT_DATA={pointdata_count}")
        
        # Extract data sections
        points_section = re.search(
            r'(POINTS\s+\d+\s+double\n(?:.*?\n)*?)(?=CELLS)',
            content,
            re.DOTALL
        )
        cells_section = re.search(
            r'(CELLS\s+.*?)(?=CELL_TYPES)',
            content,
            re.DOTALL
        )
        cell_types_section = re.search(
            r'(CELL_TYPES\s+\d+\n(?:.*?\n)*?)(?=POINT_DATA)',
            content,
            re.DOTALL
        )
        point_data_section = re.search(
            r'(POINT_DATA.*)',
            content,
            re.DOTALL
        )
        
        if not all([points_section, cells_section, cell_types_section, point_data_section]):
            print("Error: Could not parse VTK sections")
            return False
        
        # Extract all data values from POINT_DATA
        pd_text = point_data_section.group(1)
        lines = pd_text.split('\n')
        
        # Skip header lines: "POINT_DATA N", "SCALARS ...", "LOOKUP_TABLE default"
        data_start = 0
        for i, line in enumerate(lines):
            if line.startswith('LOOKUP_TABLE'):
                data_start = i + 1
                break
        
        data_values = []
        for line in lines[data_start:]:
            line = line.strip()
            if line:
                try:
                    data_values.append(float(line))
                except ValueError:
                    pass
        
        print(f"  Found {len(data_values)} data values, need {points_count}")
        
        if len(data_values) < points_count:
            print(f"  Error: Not enough data values")
            return False
        
        # Truncate to exact number of points needed
        data_values = data_values[:points_count]
        
        # Reconstruct the file
        fixed_content = content[:points_match.start()]
        fixed_content += f"POINTS {points_count} double\n"
        
        # Keep original points section
        points_lines = points_section.group(1).split('\n')
        fixed_content += '\n'.join(points_lines[1:])  # Skip "POINTS N double" line
        
        # Keep cells and cell_types
        fixed_content += cells_section.group(1)
        fixed_content += cell_types_section.group(1)
        
        # Write corrected POINT_DATA
        fixed_content += f"POINT_DATA {points_count}\n"
        fixed_content += "SCALARS dof_component_0 double 1\n"
        fixed_content += "LOOKUP_TABLE default\n"
        for val in data_values:
            fixed_content += f"{val}\n"
        
        with open(output_file, 'w') as f:
            f.write(fixed_content)
        
        print(f"✓ Fixed and wrote to {output_file.name}")
        return True
        
    except Exception as e:
        print(f"Error processing {input_file}: {e}")
        return False

def main():
    input_dir = Path("/Users/riccardocapellupo/Desktop/BGCE/gpu-amr/build/vtk_output")
    output_dir = Path("/Users/riccardocapellupo/Desktop/BGCE/gpu-amr/build/vtk_output_fixed")
    output_dir.mkdir(exist_ok=True)
    
    vtk_files = list(input_dir.glob('dg_tree_advanced_timestep_Order2_*.vtk'))
    
    if not vtk_files:
        print(f"No VTK files found in {input_dir}")
        return
    
    print(f"Found {len(vtk_files)} VTK files\n")
    
    fixed_count = 0
    for vtk_file in sorted(vtk_files):
        output_file = output_dir / vtk_file.name
        if fix_vtk_file(vtk_file, output_file):
            fixed_count += 1
    
    print(f"\n✓ Fixed {fixed_count}/{len(vtk_files)} files")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    main()
