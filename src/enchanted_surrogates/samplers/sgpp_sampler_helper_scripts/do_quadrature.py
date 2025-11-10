import os
import sys
import io
import numpy as np
import pysgpp
import faulthandler

faulthandler.enable()

if __name__ == "__main__":
    DELIMITER = b'\x00\xff\x00\xff\xfe\xfd\xfc\xfbUNLIKELY_DELIMITER\xfb\xfc\xfd\xfe\xff\x00\xff\x00'

    try:
        data = sys.stdin.buffer.read()
        if not data:
            raise ValueError("No data received on stdin")

        parts = data.split(DELIMITER)
        if len(parts) != 2:
            raise ValueError(f"Expected 2 parts after split, got {len(parts)}")

        alpha_numpy_serial, grid_serial = parts
        
        grid_str = grid_serial.decode('utf-8')
        
        # load numpy array
        alpha_np = np.load(io.BytesIO(alpha_numpy_serial))

        # convert and unserialize
        alpha = pysgpp.DataVector(alpha_np.tolist())
        grid = pysgpp.Grid.unserialize(grid_str)

        quad = pysgpp.createOperationQuadrature(grid)
        result = quad.doQuadrature(alpha)

        # print numeric result and flush
        print(float(result))
        sys.stdout.flush()

    except Exception as e:
        # report error text to stderr for parent to inspect
        print(f"ERROR: {e}", file=sys.stderr)
        sys.stderr.flush()
        # exit with nonzero code so parent can detect failure
        sys.exit(1)
