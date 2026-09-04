# Quality release gate

Every release publishes a versioned report containing extraction
precision/recall, evidence coverage, temporal accuracy, replay mismatches and
resource metrics. Threshold failures block release. Any semantic mismatch or
non-deterministic replay is blocking regardless of resource performance.
