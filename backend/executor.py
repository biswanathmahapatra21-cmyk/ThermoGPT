"""
backend/executor.py
---------------------------------------
Safe numeric code runner for ThermalGPT.
Executes short Python snippets in an isolated subprocess.
"""

import subprocess
import tempfile
import json
import textwrap


def run_calculation(code_str: str, timeout: int = 5):
    """
    Runs small Python code safely in a subprocess.
    The code must print its result as JSON.
    """

    # Cleanly format code into a standalone runnable script
    wrapped_code = (
        "import json\n"
        "try:\n"
        f"{textwrap.indent(code_str, '    ')}\n"
        "except Exception as e:\n"
        "    print(json.dumps({'error': str(e)}))\n"
    )

    # Write to a temporary file
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(wrapped_code)
        fname = f.name

    # Run in separate process
    proc = subprocess.run(
        ["python", fname],
        capture_output=True,
        text=True,
        timeout=timeout
    )

    return proc.stdout.strip(), proc.stderr.strip()


if __name__ == "__main__":
    # Test: Q = m * Cp * (Tout - Tin)
    code = "result = {'Q': 2 * 4186 * (60 - 20)}\nprint(json.dumps(result))"
    out, err = run_calculation(code)
    print('Output:', out)
    if err:
        print('Error:', err)


