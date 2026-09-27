# content.frame-asset.v1 (private, read-only)

Both routes require the configured service Bearer token and an allowed
`X-Caller-Service`. Callers supply only immutable IDs, never a path or URL.

* `GET /api/v1/content-snapshots/{content_snapshot_id}/frames/{frame_id}` returns
  `application/json` with `{"contract_version":"content.frame-asset.v1","data":{...}}`.
  `data` contains `frame_id` (string), `timestamp_ms` (integer), `image_hash`
  (lowercase SHA-256 hex), and `scopes` (array). Every scope has its own
  `knowledge_id`, `evidence_window_id`, `relation`, `ocr` and `vision`. The
  latter arrays contain `{summary,confidence}` objects; confidence may be null.
  A frame shared by several knowledge items has several scopes. Relations are
  sealed item-level visual decisions, not inferred from video-wide entities.
* `GET /api/v1/content-snapshots/{content_snapshot_id}/frames/{frame_id}/image`
  returns the exact verified image bytes with `image/jpeg` or `image/png`.

Successful responses use `200`, `Cache-Control: no-store` and
`X-Content-Type-Options: nosniff`. Authentication failures use `401` or `403`.
Unknown snapshot/frame uses `404`; incomplete or changed sealed lineage,
unsafe/missing storage, or image digest mismatch uses `409`. Errors use the
standard redacted error envelope and do not expose local paths. There are no
fixture, unsigned, public, or signed-URL reads. The asset must be a snapshot
`frames:*` member, and its item-level visual packet must be a snapshot member.
Each scope is checked against a sealed occurrence identity and one unique
frame/window/semantic-segment crosscheck; OCR and vision entries must match
their separately sealed artifact identities, hashes, frame, window, summary,
confidence and model. A producer-sealed `UNKNOWN`/`GAP` with
`CROSSCHECK_MISSING` or `CROSSCHECK_SCOPE_AMBIGUOUS` may lack a unique scoped
crosscheck, including one GAP window inside an otherwise human-review packet;
it is never promoted to support. Other missing scope evidence returns `409`,
not an inferred relation. Images are read through one bounded handle (maximum 20 MiB) and
the returned bytes are the bytes whose SHA-256 was checked.
The active snapshot must also contain a verified transcript: its media ID,
the semantic segment artifact's transcript ID/parent, the crosscheck's
transcript ID/parents, and the occurrence artifact's semantic parent must
agree with the active sealed chain. Each occurrence row must independently
name that snapshot's active source and transcript artifact IDs.

Deployment must set `CONTENT_RAW_STORAGE_DIR` to the same absolute durable root
used by the frame-producing worker (`raw_storage_dir` in its pipeline context).
The read API accepts only files directly under that root's `frames` directory.
The deterministic tests do not establish SQL repository or deployed-worker
roundtrip compatibility; that remains an environment gate.
