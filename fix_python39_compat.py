"""
Fix Python 3.9 compatibility by replacing new-style type hints with typing.Union

This script patches the needle codebase to use Python 3.9-compatible type hints.

Changes:
- tuple[int, ...] | None  →  Optional[Tuple[int, ...]]
- list[int] | None        →  Optional[List[int]]
- int | tuple             →  Union[int, tuple]

Run this before testing on Python 3.9:
    python3 fix_python39_compat.py
"""

import os
import re

files_to_fix = [
    "python/needle/backend_ndarray/ndarray.py",
    "python/needle/ops/ops_mathematic.py",
    "python/needle/nn/nn_basic.py",
    "python/needle/autograd.py",
]

def fix_type_hints(content):
    """
    Replace Python 3.10+ type hints with Python 3.9-compatible versions.
    """
    # Add typing imports if not present
    if "from typing import" not in content and ("tuple[" in content or "list[" in content or " | " in content):
        # Find the import section
        lines = content.split('\n')
        import_idx = 0
        for i, line in enumerate(lines):
            if line.startswith('import ') or line.startswith('from '):
                import_idx = i + 1

        # Insert typing import
        lines.insert(import_idx, "from typing import Optional, Union, Tuple, List")
        content = '\n'.join(lines)

    # Replace type hints
    # Pattern 1: tuple[int, ...] | None  →  Optional[Tuple[int, ...]]
    content = re.sub(r'tuple\[int, \.\.\.\] \| None', 'Optional[Tuple[int, ...]]', content)

    # Pattern 2: list[int] | None  →  Optional[List[int]]
    content = re.sub(r'list\[int\] \| None', 'Optional[List[int]]', content)

    # Pattern 3: int | tuple[int, ...] | list[int] | None  →  Optional[Union[int, Tuple[int, ...], List[int]]]
    content = re.sub(
        r'int \| tuple\[int, \.\.\.\] \| list\[int\] \| None',
        'Optional[Union[int, Tuple[int, ...], List[int]]]',
        content
    )

    # Pattern 4: int | tuple[int, ...]  →  Union[int, Tuple[int, ...]]
    content = re.sub(r'int \| tuple\[int, \.\.\.\]', 'Union[int, Tuple[int, ...]]', content)

    # Pattern 5: tuple[int, ...] | None (in function params)
    content = re.sub(r'axes: tuple\[int, \.\.\.\] \| None', 'axes: Optional[Tuple[int, ...]]', content)

    return content


def main():
    print("Fixing Python 3.9 compatibility...")
    print("=" * 80)

    for filepath in files_to_fix:
        if not os.path.exists(filepath):
            print(f"⚠ Skipping {filepath} (not found)")
            continue

        print(f"\nProcessing {filepath}...")

        # Read file
        with open(filepath, 'r') as f:
            original_content = f.content()

        # Fix type hints
        fixed_content = fix_type_hints(original_content)

        # Check if changes were made
        if original_content != fixed_content:
            # Create backup
            backup_path = filepath + '.py39bak'
            with open(backup_path, 'w') as f:
                f.write(original_content)
            print(f"  ✓ Created backup: {backup_path}")

            # Write fixed version
            with open(filepath, 'w') as f:
                f.write(fixed_content)
            print(f"  ✓ Fixed type hints")

            # Show changes
            changes = sum(1 for a, b in zip(original_content.split('\n'), fixed_content.split('\n')) if a != b)
            print(f"  ✓ Modified {changes} lines")
        else:
            print(f"  - No changes needed")

    print("\n" + "=" * 80)
    print("✓ Python 3.9 compatibility fixes applied")
    print("\nTo restore original files:")
    print("  for f in python/needle/**/*.py39bak; do mv $f ${f%.py39bak}; done")
    print("\nNow you can run:")
    print("  python3 test_freq_conv_all_backends.py")


if __name__ == "__main__":
    main()
