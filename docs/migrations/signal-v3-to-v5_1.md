# Migration guide: v3 to v5.1

Run compatibility and formal readers in parallel, then switch formal consumers
to `content-factor-signal.v5.1`. Formal requests must bind business,
knowledge and availability clocks plus `content_snapshot_id`; compatibility
responses remain non-formal and carry a sunset date.
