"""Canonical labeling for periodic graph isomorphism testing.

Used by contrib/ for duplicate detection on user submissions.

Note: The brute-force enumeration in this module is exposed only for
its canonical-labeling helpers. The enumeration itself has known
limitations (overcounts at small unit cell sizes vs Irvine 2016
Table 5.2) and should not be relied on as a complete enumerator.
"""
