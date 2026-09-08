# Authorized video sources

The video worker may process public Bilibili pages and operator-authorized,
non-DRM Xiaoe material only.  It does not bypass login, paywalls, regional
restrictions, or DRM.

For an authorized Bilibili page, prefer a read-only, operator-created
Playwright storage state: configure `CONTENT_BILIBILI_STORAGE_STATE_REF` and
`CONTENT_BILIBILI_STORAGE_STATE_FILE`, then submit that reference as
`credential_ref`.  The worker validates the state and converts only in-scope,
unexpired Bilibili cookies into a private, short-lived Netscape cookie file for
one yt-dlp resolution.  It removes that file in `finally`; state/cookie values
never enter task rows, checkpoints, artifacts, reports, errors, or logs. The
login helper uses an isolated visible Playwright browser and never reads an
existing Chrome profile.

To create a state, install the `media-browser` extra and run the helper on the
operator workstation. It opens a fresh headed browser for at most the supplied
window. After completing login, return to the terminal and press Enter at
`READY_FOR_LOGIN`; only then does it check and save an allowed-domain cookie.
Enter the password and any OTP yourself in that browser; never put either in a
command, environment variable, task payload, or support request. The command
prints only a success or safe error code, never a state path, cookie, or hash.

```powershell
stock-content-capture-storage-state --page-url https://www.bilibili.com/ --allowed-domain bilibili.com --allowed-domain bilivideo.com --destination D:\secrets\bilibili-state.json --timeout-seconds 300
stock-content-capture-storage-state --page-url https://m.xiaoe-tech.com/ --allowed-domain xiaoe-tech.com --destination D:\secrets\xiaoe-state.json --timeout-seconds 300
```

If the browser is closed, the time expires, or the login yields no cookie for
one of the exact allowed domains, no state is saved. On POSIX the saved state
is `0400`; on Windows it has an explicit non-inherited ACL granting only the
current user (read) and LocalSystem (full control). Store and mount it as an
operator-managed secret file.

`CONTENT_BILIBILI_CREDENTIAL_REF` together with
`CONTENT_BILIBILI_COOKIEFILE` remains the legacy read-only Netscape-cookie
configuration.  When both configured references could match, storage state
has priority. Public Bilibili requests omit `credential_ref` and continue to
resolve without login.

For signed Xiaoe HLS, mount a read-only file containing the current authorized
locator and configure `CONTENT_XIAOE_HLS_CREDENTIAL_REF` and
`CONTENT_XIAOE_HLS_LOCATOR_FILE`.  Submit a query-free public `.m3u8` identity
plus that reference.  The worker reads the signed locator only at execution
and resolves it again once if materialization reports expiry.  Task rows,
checkpoints, artifacts, and error messages contain neither the signed URL nor
the reference name.

`docker compose --profile video` consumes the corresponding host-file
variables: `CONTENT_BILIBILI_STORAGE_STATE_HOST_FILE`,
`CONTENT_BILIBILI_COOKIEFILE_HOST_FILE`, and
`CONTENT_XIAOE_HLS_LOCATOR_HOST_FILE`.  These files must be operator-managed
and read-only; live source access is intentionally outside automated tests.
