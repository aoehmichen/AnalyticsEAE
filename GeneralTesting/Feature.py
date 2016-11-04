__author__ = 'Axel Oehmichen'

__all__ = ['Feature', 'Normality', 'Summary']

familyList = ['Binary', 'Numerical', 'Categorical', 'Other']


def isFloat(f):
    try:
        i = 0
        if str(f[i]).lower() in ("unknown", "uncertain"):
            i += 1

        float(f[i])
        # Special treatment for cases where we have 1 and 0 which becomes binary and not float
        if float(f[i]) == 1.0 or float(f[i]) == 0.0:
            for j in range(len(f)):
                if float(f[j]) not in (0.0, 1.0):
                    return "Float"
            return "Bool"
        else:
            return "Float"
    except ValueError:
        pass

    try:
        import unicodedata

        unicodedata.numeric(f)
        return "Float"
    except (TypeError, ValueError):
        pass

    return "Other"


def isBool(b):
    i = 0
    while b[i].lower() in ("uncertain", "unknown"):
        i += 1
    return b[i].lower() in ("yes", "no", "1", "0", "1.0", "0.0", "present", "absent", "male", "female", "true", "t", "false", "f")


def isGarbage(f):
    if len(f) < 4:
        return True
    classes = list(set([str(x).lower() for x in f]))
    if len(classes) < 2:
        return True
    return False

def _inferFamily(featureValues):
    if isGarbage(featureValues):
        return "Other"

    value = isFloat(featureValues)
    if value is "Float":
        return "Numerical"
    elif value is "Bool":
        return "Binary"

    if isBool(featureValues):
        return "Binary"
    else:
        return "Categorical"

def _putTrueType(featureValues):
    family = _inferFamily(featureValues)
    if family is "Numerical":
        value = [float(x) for x in featureValues]
        return value
    elif family in ("Binary", "Categorical"):
        value = [str(x) for x in featureValues]
        return value
    else:
        return featureValues


class Feature(object):
    def __init__(self, featureName, featureValues, originalValues):
        self.featureName = str(featureName)
        self.featureValues = _putTrueType(featureValues)
        self.originalValues = originalValues
        self.featureFamily = _inferFamily(featureValues)
        self.normality = Normality()
        self.distribution = "NotComputed"  # name of the distribution

    def __reduce__(self):
        return (Feature, (self.featureName, self.featureValues, self.originalValues))

    def __str__(self):
        return "The feature is : " + str(self.featureName) + "\n" + "The feature Value is : " + ', '.join(self.originalValues)


class Normality(object):

    def __init__(self):
        self.anderson = None
        self.shapiro = None

    def __str__(self):
        andersonValues = "Anderson test : " + str(self.anderson) + "\n"
        shapiroValues = "Shapiro test : " + str(self.shapiro)
        return andersonValues + shapiroValues

class Summary(object):
    def __init__(self, numberOfNas):
        self.numberOfNas = numberOfNas
        #self.features = _convert_to_dictionnary(featuresNames, featuresValues)

