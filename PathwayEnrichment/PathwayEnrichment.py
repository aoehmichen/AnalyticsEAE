import sys

from numpy.core.multiarray import ndarray
from pyspark import SparkConf, SparkContext
from scipy.stats import fisher_exact
from PE_DataFromKEGG import Kegg, Bonferroni, HolmBonferroni, Sidak

from datetime import datetime
from bson.objectid import ObjectId
from pymongo import MongoClient

__author__ = 'Axel Oehmichen'

__all__ = ['PathwayEnrichment']

#####################################
# functions                         #
#####################################

def uniqifyList(seq):
    # Not order preserving
    return {}.fromkeys(seq).keys()

def convertSymbolsToIDs(listOfGenesSymbols, mapping):
    listOfGeneswithIDs = []
    for i in range(len(listOfGenesSymbols)):
        if listOfGenesSymbols[i] in mapping.keys():
            listOfGeneswithIDs.append(mapping[listOfGenesSymbols[i]])
    return listOfGeneswithIDs


def _buildTable(pathway, ListOfGenes):
    table = ndarray(shape=(2, 2), dtype=int, order='F')

    lenListOfGenes = len(ListOfGenes)
    lenPathway = len(pathway)

    numberOfElementsInPathwayAndList = len([i for i in ListOfGenes if i in pathway])

    table[0][0] = numberOfElementsInPathwayAndList
    table[0][1] = lenListOfGenes - numberOfElementsInPathwayAndList
    table[1][0] = lenPathway - numberOfElementsInPathwayAndList
    table[1][1] = Kegg.NUMBER_OF_GENES - (lenPathway + lenListOfGenes - numberOfElementsInPathwayAndList)

    return table

# The method executing fisher's exact test
def _enrichment(pathway, ListOfGenes):
    table = _buildTable(pathway[1], ListOfGenes)

    alternative = 'two-sided'
    fisherRes = fisher_exact(table, alternative)

    return (pathway[0], fisherRes)


#####################################
# Pathway Enrichment                #
#####################################
class PathwayEnrichment(object):
    def __init__(self, sc, listOfGenes):
        self.pathwayDatabases = "KEGG"
        self.listofgenes = listOfGenes
        self.enrichement = self._do_enrichment(sc, listOfGenes)

    def __reduce__(self):
        return (PathwayEnrichment, (self.pathwayDatabases, []))

    def __str__(self):
        return "The list of genes is : " + str(self.listofgenes)

    def _do_enrichment(self, sc, listOfGenes):
        if not listOfGenes:
            return []

        # We retrive the file containing the list of genes for each pathway.
        pathwaysTemp = sc.textFile("Link_pathways_genes.txt") \
            .map(lambda l: l.split('\t')) \
            .map(lambda l: (str((l[0].split(':'))[1]), int((l[1].split(':'))[1]))) \
            .groupByKey()

        pathways = pathwaysTemp.map(lambda l: (l[0], list(l[1])))
        enrichmentRes = pathways.map(lambda pathway: _enrichment(pathway, listOfGenes))

        return enrichmentRes

    @staticmethod
    def doSparkEnrichment(sc, listOfGenesSymbols, correction):

        # We make sure that the listOfGenesSymbols contains only unique Ids. This is needed when we have several probes
        # for one gene
        ListOfUniqueGenesSybols = uniqifyList(listOfGenesSymbols)

        # We load the mapping file to transform NCBI's Gene symbols into Gene Ids
        mappingFile = sc.textFile("pe_genes.txt").map(lambda p: p.split('\t')).collect()
        mapping = dict()
        for tuple in mappingFile:
            if tuple[1]:
                mapping[str(tuple[0])] = int(tuple[1])

        # Transform ou gene symbols into Ids
        listOfGenes = convertSymbolsToIDs(ListOfUniqueGenesSybols, mapping)
        pathwayEnrichment = PathwayEnrichment(sc, listOfGenes)

        fishers = pathwayEnrichment.enrichement
        top5 = fishers.takeOrdered(5, lambda s: s[1][1])

        # We do the multiple corrections
        pValues = [pVal[1][1] for pVal in top5]
        corrected_pValues = pValues
        if correction in 'Bonferroni':
            corrected_pValues = Bonferroni(pValues, a=0.05).corrected_pvals
        elif correction in 'HB':
            corrected_pValues = HolmBonferroni(pValues, a=0.05).corrected_pvals
        elif correction in 'Sidak':
            corrected_pValues = Sidak(pValues, a=0.05).corrected_pvals

        for i in range(len(top5)):
            top5[i] = (top5[i][0], corrected_pValues[i])

        resp = Kegg.retrievePathwayOrGene(top5[0][0])

        return top5, resp, sorted(ListOfUniqueGenesSybols, reverse=True), listOfGenes

#####################################
# main program                      #
#####################################

if __name__ == "__main__":
    conf = SparkConf().setAppName("PathwayEnrichment")
    sc = SparkContext(conf=conf)

    # We check that the arguments are well formed
    if not len(sys.argv) == 4:
        print >> sys.stderr, \
            "Invalid number of arguments. Usage: PathwayEnrichment.py <CorrectionToUse, String> <mongoIP, String> <mongoDocId, String>"
        exit(-1)

    # We load the list genes Symbols
    correction = str(sys.argv[1])
    mongoIP = str(sys.argv[2])
    mongoDocId = str(sys.argv[3])

    ## we retrieve the mongo client and database for Pathway enrichment
    db = MongoClient('mongodb://' + mongoIP +'/').eae
    db.authenticate('eae', 'eae', mechanism='SCRAM-SHA-1')
    peCollection = db.PathwayEnrichment
    docId = ObjectId(mongoDocId)
    document = peCollection.find_one({'_id': docId})

    listOfGenesSymbols = [str(x) for x in document["CustomField"].split(' ')]

    top5, resp, listOfGenesSymbols, listOfGenes = PathwayEnrichment.doSparkEnrichment(sc, listOfGenesSymbols, correction)

    doc = {"TopPathways": top5,
           "KeggTopPathway": resp.text,
           "ListOfGenes": ' '.join(listOfGenesSymbols),
           "ListOfGenesIDs": ' '.join([str(x) for x in listOfGenes]),
           "EndTime": datetime.now(),
           "Status": "Completed"}

    peCollection.update_one({'_id': docId}, {"$set": doc}, upsert=False)
