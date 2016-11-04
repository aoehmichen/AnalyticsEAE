import numpy as np
import requests
import sys

__author__ = 'Axel Oehmichen'

__all__ = ['Kegg', 'Bonferroni', 'HolmBonferroni', 'Sidak']

class Kegg(object):

    KEGG_REST_URL = "http://rest.kegg.jp/"
    NUMBER_OF_GENES = 30922 # See http://rest.kegg.jp/info/hsa to see where this number comes from

    def __init__(self):
        self.KEGG_REST_URL = "http://rest.kegg.jp/"

    @classmethod
    def retrievePathwayOrGene(cls, name):
        resp = requests.get(cls.KEGG_REST_URL +"get/" + name)

        if(resp.status_code != 200):
            print >> sys.stderr, "The request is not well formed. The HTTP status code is : %i" %resp.status_code
            exit(-1)

        return resp

    @classmethod
    def retrievePathwayOrGeneImage(cls, name):
        resp = requests.get(cls.KEGG_REST_URL +"get/" + name + "/image")

        if(resp.status_code != 200):
            print >> sys.stderr, "The request is not well formed. The HTTP status code is : %i" %resp.status_code
            exit(-1)

        return resp

    @classmethod
    def convNCBItoKEGG(cls):
        resp = requests.get(cls.KEGG_REST_URL +"conv/hsa/ncbi-geneid")
        if(resp.status_code != 200):
            print >> sys.stderr, "The request is not well formed. The HTTP status code is : %i" %resp.status_code
            exit(-1)

        return resp


class AbstractCorrection(object):

    def __init__(self, pvals, a=.05):
        self.pvals = self.corrected_pvals = np.array(pvals)
        self.n = len(self.pvals)    # number of multiple tests
        self.a = a                  # type-1 error cutoff for each test

        self.set_correction()
        # Reset all pvals > 1 to 1
        self.corrected_pvals[self.corrected_pvals > 1] = 1

    def set_correction(self):
        # the purpose of multiple correction is to lower the alpha
        # instead of the canonical value (like .05)
        pass

class Bonferroni(AbstractCorrection):

    def set_correction(self):
        self.corrected_pvals *= self.n


class Sidak(AbstractCorrection):
    def set_correction(self):
        if self.n != 0:
            correction = self.a * 1. / (1 - (1 - self.a) ** (1. / self.n))
        else:
            correction = 1
        self.corrected_pvals *= correction


class HolmBonferroni(AbstractCorrection):

    def set_correction(self):
        if len(self.pvals):
            idxs, correction = list(zip(*self.generate_significant()))
            idxs = list(idxs)
            self.corrected_pvals[idxs] *= correction

    def generate_significant(self):

        pvals = self.pvals
        pvals_idxs = list(zip(pvals, list(range(len(pvals)))))
        pvals_idxs.sort()

        lp = len(self.pvals)

        from itertools import groupby
        for pval, idxs in groupby(pvals_idxs, lambda x: x[0]):
            idxs = list(idxs)
            for p, i in idxs:
                if p * 1. / lp < self.a:
                    yield (i, lp)
            lp -= len(idxs)
