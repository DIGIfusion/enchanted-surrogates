import os
import sys
import io
import numpy as np
import pysgpp
import faulthandler
import traceback
faulthandler.enable()

if __name__ == "__main__":
    DELIMITER = b'\x00\xff\x00\xff\xfe\xfd\xfc\xfbUNLIKELY_DELIMITER\xfb\xfc\xfd\xfe\xff\x00\xff\x00'

    try:
        data = sys.stdin.buffer.read()
        if not data:
            raise ValueError("No data received on stdin")

        parts = data.split(DELIMITER)
        if len(parts) != 3:
            raise ValueError(f"Expected 3 parts after split, got {len(parts)}")

        alpha_numpy_serial, grid_serial, unit_points_serial = parts

        grid_str = grid_serial.decode('utf-8')
        
        # load numpy array
        alpha_np = np.load(io.BytesIO(alpha_numpy_serial))
        unit_points_np = np.load(io.BytesIO(unit_points_serial))
        
        # convert and unserialize
        alpha = pysgpp.DataVector(alpha_np.tolist())
        grid = pysgpp.Grid.unserialize(grid_str)
        # sys.stderr.write(f"debug unit_points_np shape {unit_points_np.shape} array {unit_points_np}\n")
        # sys.stderr.flush()
        unit_points_dm = pysgpp.DataMatrix(unit_points_np.tolist())
        
        opEval = pysgpp.createOperationMultipleEval(grid, unit_points_dm)
        results = pysgpp.DataVector(len(unit_points_np))
        opEval.eval(alpha, results)
        #results.array() # for conversion to numpy but sometimes this line breaks python. 
        result = np.array([results[i] for i in range(results.getSize())])

        # write binary .npy to stdout.buffer
        buf = io.BytesIO()
        np.save(buf, result)
        buf.seek(0)
        sys.stdout.buffer.write(buf.getvalue())
        sys.stdout.buffer.flush()
        
    except Exception as e:
        # report error text to stderr for parent to inspect
        print(f"predict.py ERROR: {e}", file=sys.stderr)
        print(f"predict.py TRACEBACK: {traceback.format_exc()}", file=sys.stderr)
        sys.stderr.flush()
        # exit with nonzero code so parent can detect failure
        sys.exit(1)
