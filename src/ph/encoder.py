import json
import cupy as cp

class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (cp.int_, cp.intc, cp.intp, cp.int8,
                            cp.int16, cp.int32, cp.int64, cp.uint8,
                            cp.uint16, cp.uint32, cp.uint64)):

            return int(obj)

        elif isinstance(obj, (cp.float_, cp.float16, cp.float32, cp.float64)):
            return float(obj)

        elif isinstance(obj, (cp.complex_, cp.complex64, cp.complex128)):
            return {'real': obj.real, 'imag': obj.imag}

        elif isinstance(obj, (cp.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (cp.bool_)):
            return bool(obj)

        elif isinstance(obj, (cp.void)): 
            return None

        return json.JSONEncoder.default(self, obj)