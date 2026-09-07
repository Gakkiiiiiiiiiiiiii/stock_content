# Authorized video sources

The video worker may process public Bilibili pages and operator-authorized,
non-DRM Xiaoe material only.  It does not bypass login, paywalls, regional
restrictions, or DRM.

For an authorized Bilibili page, mount a read-only Netscape cookie file and
configure `CONTENT_BILIBILI_CREDENTIAL_REF` together with
`CONTENT_BILIBILI_COOKIEFILE`.  Submit the configured reference as
`credential_ref`; never submit cookie bytes.  The worker resolves the
reference from its local allowlist and supplies the file only to Bilibili
tooling.  Public Bilibili requests omit the reference.

For signed Xiaoe HLS, mount a read-only file containing the current authorized
locator and configure `CONTENT_XIAOE_HLS_CREDENTIAL_REF` and
`CONTENT_XIAOE_HLS_LOCATOR_FILE`.  Submit a query-free public `.m3u8` identity
plus that reference.  The worker reads the signed locator only at execution
and resolves it again once if materialization reports expiry.  Task rows,
checkpoints, artifacts, and error messages contain neither the signed URL nor
the reference name.

`docker compose --profile video` consumes the corresponding host-file
variables: `CONTENT_BILIBILI_COOKIEFILE_HOST_FILE` and
`CONTENT_XIAOE_HLS_LOCATOR_HOST_FILE`.  These files must be operator-managed
and read-only; live source access is intentionally outside automated tests.
