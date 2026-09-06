# Quality release gate

Every release publishes a versioned report containing extraction
precision/recall, evidence coverage, temporal accuracy, replay mismatches and
resource metrics. Threshold failures block release. Any semantic mismatch or
non-deterministic replay is blocking regardless of resource performance.

`config/quality-gates.json` is the auditable source for the fixed small-fixture
benchmark and the core/media/multimodal worker CPU and memory limits. The CI
quality gate verifies each profile SBOM against its reviewed lock, validates
those compose limits, and stores the benchmark observation as an artifact.
The deterministic fixture preserves the three-clock replay checks; a resource
result never overrides a semantic or replay failure.
