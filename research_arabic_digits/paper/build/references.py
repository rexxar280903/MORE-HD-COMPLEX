"""Basis data referensi paper MADBase (salinan dari paper/build/references.py studi MNIST + referensi MADBase).
Reference database. Every DOI entry was verified against the Crossref API on 2026-10-06
(crossref_refs.json holds the retrieved metadata); arXiv entries were verified against the arXiv API.
Entries with a DOI are formatted automatically from the Crossref record (authors, title, volume, year,
pages); "journal" overrides the Crossref short title with the standard abbreviation.
"""

REFS = {
    # --- MORE and multi-class VQC -------------------------------------------------------------
    "wu2023": {"text": "J. Wu, T. Hu and Q. Li, MORE: Measurement and correlation based variational quantum "
                       "circuit for multi-classification, in Proc. 2023 IEEE Int. Conf. Quantum Computing and "
                       "Engineering (QCE) (IEEE, Bellevue, 2023), pp. 208–218, doi:10.1109/QCE57702.2023.00031."},
    "perez2020": {"doi": "10.22331/q-2020-02-06-226", "journal": "Quantum"},
    "lloyd2020": {"text": "S. Lloyd, M. Schuld, A. Ijaz, J. Izaac and N. Killoran, Quantum embeddings for machine "
                          "learning, preprint (2020), arXiv:2001.03622."},
    "cong2019": {"doi": "10.1038/s41567-019-0648-8", "journal": "Nat. Phys."},
    "grant2018": {"doi": "10.1038/s41534-018-0116-9", "journal": "npj Quantum Inf."},
    # --- QML background ----------------------------------------------------------------------
    "biamonte2017": {"doi": "10.1038/nature23474", "journal": "Nature"},
    "preskill2018": {"doi": "10.22331/q-2018-08-06-79", "journal": "Quantum",
                     "title": "Quantum computing in the NISQ era and beyond"},
    "bharti2022": {"doi": "10.1103/RevModPhys.94.015004", "journal": "Rev. Mod. Phys."},
    "cerezo2021": {"doi": "10.1038/s42254-021-00348-9", "journal": "Nat. Rev. Phys."},
    "benedetti2019": {"doi": "10.1088/2058-9565/ab4eb5", "journal": "Quantum Sci. Technol."},
    "mitarai2018": {"doi": "10.1103/PhysRevA.98.032309", "journal": "Phys. Rev. A"},
    "farhi2018": {"text": "E. Farhi and H. Neven, Classification with quantum neural networks on near term "
                          "processors, preprint (2018), arXiv:1802.06002."},
    "schuld2020": {"doi": "10.1103/PhysRevA.101.032308", "journal": "Phys. Rev. A"},
    "havlicek2019": {"doi": "10.1038/s41586-019-0980-2", "journal": "Nature"},
    # --- expressibility / trainability ----------------------------------------------------
    "mcclean2018": {"doi": "10.1038/s41467-018-07090-4", "journal": "Nat. Commun."},
    "holmes2022": {"doi": "10.1103/PRXQuantum.3.010313", "journal": "PRX Quantum",
                   "title": "Connecting ansatz expressibility to gradient magnitudes and barren plateaus"},
    "sim2019": {"doi": "10.1002/qute.201900070", "journal": "Adv. Quantum Technol.",
                "title": "Expressibility and entangling capability of parameterized quantum circuits for hybrid "
                         "quantum-classical algorithms"},
    "schuld2021": {"doi": "10.1103/PhysRevA.103.032430", "journal": "Phys. Rev. A"},
    "abbas2021": {"doi": "10.1038/s43588-021-00084-1", "journal": "Nat. Comput. Sci."},
    "huang2021": {"doi": "10.1038/s41467-021-22539-9", "journal": "Nat. Commun."},
    "cerezo2022": {"doi": "10.1038/s43588-022-00311-3", "journal": "Nat. Comput. Sci."},
    "schuldkilloran2022": {"doi": "10.1103/PRXQuantum.3.030101", "journal": "PRX Quantum",
                           "title": "Is quantum advantage the right goal for quantum machine learning?"},
    "renou2021": {"doi": "10.1038/s41586-021-04160-4", "journal": "Nature"},
    # --- features and data ------------------------------------------------------------------
    "hu1962": {"doi": "10.1109/TIT.1962.1057692", "journal": "IRE Trans. Inf. Theory", "authors": "M.-K. Hu"},
    "teague1980": {"doi": "10.1364/JOSA.70.000920", "journal": "J. Opt. Soc. Am.",
                   "title": "Image analysis via the general theory of moments"},
    "khotanzad1990": {"doi": "10.1109/34.55109", "journal": "IEEE Trans. Pattern Anal. Mach. Intell."},
    "lecun1998": {"doi": "10.1109/5.726791", "journal": "Proc. IEEE",
                  "authors": "Y. LeCun, L. Bottou, Y. Bengio and P. Haffner"},
    # --- MADBase (verified against Crossref API 2026-10-11; official page accessed 2026-10-08) -------
    "abdleazeem2008": {"doi": "10.1007/s10032-008-0073-5", "journal": "Int. J. Doc. Anal. Recognit."},
    "madbaseweb": {"text": "S. Abdelazeem and E. El-Sherif, The Arabic handwritten digits databases ADBase & MADBase, American University in Cairo, https://datacenter.aucegypt.edu/shazeem/ (accessed 8 October 2026)."},
    # --- optimisation and software ---------------------------------------------------------
    "powell1994": {"text": "M. J. D. Powell, A direct search optimization method that models the objective and "
                           "constraint functions by linear interpolation, in Advances in Optimization and Numerical "
                           "Analysis, eds. S. Gomez and J.-P. Hennart (Springer, Dordrecht, 1994), pp. 51–67, "
                           "doi:10.1007/978-94-015-8330-5_4."},
    "virtanen2020": {"doi": "10.1038/s41592-019-0686-2", "journal": "Nat. Methods"},
    "harris2020": {"doi": "10.1038/s41586-020-2649-2", "journal": "Nature"},
    "pedregosa2011": {"text": "F. Pedregosa et al., Scikit-learn: Machine learning in Python, J. Mach. Learn. Res. "
                              "12 (2011) 2825–2830."},
    "bergholm2018": {"text": "V. Bergholm et al., PennyLane: Automatic differentiation of hybrid quantum-classical "
                             "computations, preprint (2018), arXiv:1811.04968."},
    # --- statistics -------------------------------------------------------------------------
    "mcnemar1947": {"doi": "10.1007/BF02295996", "journal": "Psychometrika",
                    "title": "Note on the sampling error of the difference between correlated proportions or "
                             "percentages"},
    "holm1979": {"text": "S. Holm, A simple sequentially rejective multiple test procedure, Scand. J. Stat. 6 "
                         "(1979) 65–70."},
    "hedges1981": {"doi": "10.3102/10769986006002107", "journal": "J. Educ. Stat.",
                   "title": "Distribution theory for Glass's estimator of effect size and related estimators"},
    # --- mechanistic analysis --------------------------------------------------------------
    "schuld2019": {"doi": "10.1103/PhysRevA.99.032331", "journal": "Phys. Rev. A"},
    "roy2007": {"text": "O. Roy and M. Vetterli, The effective rank: A measure of effective dimensionality, in Proc. "
                        "15th European Signal Processing Conf. (EUSIPCO) (Poznań, 2007), pp. 606–610, "
                        "doi:10.5281/zenodo.40328."},
    "alain2016": {"text": "G. Alain and Y. Bengio, Understanding intermediate layers using linear classifier probes, "
                          "preprint (2016), arXiv:1610.01644."},
}
