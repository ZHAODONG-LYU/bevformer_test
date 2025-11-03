#!/usr/bin/env python
"""
Script to disable PAP (Probability-Aware Point Pruning) in mmcv's multi_scale_deform_attn.py
Run this inside your docker container.
"""
import os
import re

def disable_pap():
    try:
        import mmcv
        mmcv_path = os.path.dirname(mmcv.__file__)
        target_file = os.path.join(mmcv_path, 'ops', 'multi_scale_deform_attn.py')
        
        if not os.path.exists(target_file):
            print(f"Error: File not found: {target_file}")
            return False
        
        print(f"Found mmcv file: {target_file}")
        
        # Read the file
        with open(target_file, 'r') as f:
            content = f.read()
        
        # Backup original file
        backup_file = target_file + '.backup'
        with open(backup_file, 'w') as f:
            f.write(content)
        print(f"Backup created: {backup_file}")
        
        # Pattern to match PAP code block
        # Looking for lines like:
        # pruned_mask = attention_weights < 0.01
        # if pruned_mask.any():
        #     print(...)
        #     attention_weights = attention_weights.masked_fill(pruned_mask, 0.0)
        
        # Replace PAP pruning logic
        modified = False
        lines = content.split('\n')
        new_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # Check if this line contains PAP-related code
            if 'pruned_mask' in line and 'attention_weights' in line and '<' in line:
                print(f"Found PAP code at line {i+1}: {line.strip()}")
                # Comment out this line and the next few lines
                new_lines.append('        # ' + line.strip() + '  # DISABLED BY disable_pap_in_mmcv.py')
                modified = True
                i += 1
                
                # Comment out the next lines (if block)
                while i < len(lines):
                    next_line = lines[i]
                    # Stop if we reach a line with same or less indentation (end of if block)
                    if next_line.strip() and not next_line.startswith(' ' * 12):
                        break
                    if 'masked_fill' in next_line or 'print' in next_line or 'if pruned_mask' in next_line:
                        new_lines.append('        # ' + next_line.strip() + '  # DISABLED BY disable_pap_in_mmcv.py')
                        print(f"  Commenting out line {i+1}: {next_line.strip()}")
                        i += 1
                    else:
                        i += 1
                        break
            else:
                new_lines.append(line)
                i += 1
        
        if not modified:
            print("Warning: No PAP code found. The file might have been already modified or has a different structure.")
            return False
        
        # Write modified content
        new_content = '\n'.join(new_lines)
        with open(target_file, 'w') as f:
            f.write(new_content)
        
        print(f"\nSuccessfully disabled PAP in: {target_file}")
        print(f"Original file backed up to: {backup_file}")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = disable_pap()
    if success:
        print("\n✓ PAP has been disabled successfully!")
        print("You can now run your quantization test without PAP interference.")
    else:
        print("\n✗ Failed to disable PAP. Please check the error messages above.")

