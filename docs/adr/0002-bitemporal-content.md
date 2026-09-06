# ADR 0002: Bitemporal content

Status: accepted.

Business-effective (`business_as_of`) and knowledge-recorded
(`knowledge_as_of`) times are distinct; public availability
(`availability_as_of`) is a third clock. A `content-factor-signal.v5.1`
formal request binds all three clocks and a `content_snapshot_id`; the same
bindings participate in the formal projection identity. Compatibility v3/v4/v5
responses are not substitutes for a formal v5.1 assertion.
