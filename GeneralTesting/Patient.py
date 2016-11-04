
import array
import numpy as np

from pyspark.mllib.linalg import Vector, DenseVector, SparseVector

__author__ = 'Axel Oehmichen'

__all__ = ['Patient']

# Check whether we have SciPy. MLlib works without it too, but if we have it, some methods,
# such as _dot and _serialize_double_vector, start to support scipy.sparse matrices.

try:
    import scipy.sparse
    _have_scipy = True
except:
    # No SciPy in environment, but that's okay
    _have_scipy = False


def _dictionnaries(featuresNames, featuresValues):
    dictionnary = (zip(featuresNames, featuresValues))
    return dictionnary

def _convertToDictionnary(featuresNames, l):
    if isinstance(l, Vector) and isinstance(featuresNames, Vector):
        features = _dictionnaries(featuresNames,l)
        return features
    elif type(l) in (array.array, np.array, np.ndarray, list, tuple, xrange) and type(featuresNames) in \
            (array.array, np.array, np.ndarray, list, tuple, xrange):
        transformL = DenseVector(l)
        transformNames = DenseVector(featuresNames)
        features = _dictionnaries(transformNames, transformL)
        return features
    elif _have_scipy and scipy.sparse.issparse(l):
        assert l.shape[1] == 1, "Expected column vector"
        csc = l.tocsc()
        return SparseVector(l.shape[0], csc.indices, csc.data)
    else:
        raise TypeError("Cannot convert type %s into Vector" % type(l))


class Patient(object):
    def __init__(self, patientId, featuresValues, featuresNames):
        self.patientId = str(patientId)
        self.features = _convertToDictionnary(featuresNames, featuresValues)

    def __reduce__(self):
        return (Patient, (self.patientId, self.features))

    def __str__(self):
        return "Patient id : " + str(self.patientId)
