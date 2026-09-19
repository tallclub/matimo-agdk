# Matimo Gateway server contract, as implemented

Vendored into this repository on 2026-09-18 from the build-time contract derived by reading the Universal-AgentForge (Matimo Workbench) source. File and line citations below refer to that repository, not to this one. Section numbers are what the SDK's docstrings cite.

---

# Matimo Gateway `/v1` Server Contract (code-verified, 2026-09-18)

Extracted directly from the live TypeScript on `feature/gateway`. Every claim below cites a
file:line. Design docs (`PRD.md`/`TRD.md`/`Architecture.md`/`BUILD-PLAN.md`/`TELEMETRY.md`) were
read only for intent; where they disagree with code, code wins and the disagreement is logged in
the "Drift" section at the end. AGDK must implement exactly what is described here.

All paths are relative to the router mount point. `gatewayV1Router` is mounted at `app.use('/v1',
gatewayV1Router)` (`src/backend/src/core/server.ts:258`), **before** `compression()`, the global
`express.json()`, and the global `authenticateToken` gate (`server.ts:819`). It has its own
complete auth stack , there is no session-JWT surface at all on this router
(`src/backend/src/api/v1/gateway.ts:1-24`).

Base URL for local dev: `http://localhost:8000/v1` (matches
`tests/external-agents/langchain-agent/agent.py:136`).

---

## 0. Universal request/response conventions

- **Body parsing**: this router's own raw-body-capturing `express.json()` (`gateway.ts:893-901`),
  `limit: '2mb'`, `type: () => true` (matches every Content-Type, not just `application/json` ,
  finding #9, 2026-09-13). `req.rawBody` (the exact byte buffer) is what JWS `body_hash`
  verification hashes against , never a re-`JSON.stringify()` of the parsed body.
- **Success envelope**: almost every route returns `{ data: <payload> }` (e.g. `gateway.ts:1188`,
  `:1250`, `:1309`, `:1352`, `:1429`, `:1468`, `:1520`, `:1555`, `:1590`, `:1629`). Two exceptions:
  - `GET /v1/identities/:agentId/jwks` returns a bare `{ keys: [...] }`, no `data` wrapper
    (`gateway.ts:1391`).
  - `POST /v1/chat/completions` / `POST /v1/messages` non-error responses return the raw
    OpenAI/Anthropic-shaped body directly, no `{data}` wrapper (`gateway.ts:1027`, `:1120`).
- **Error envelope** (`sendError`, `gateway.ts:254-256`): `{ error: "<code_string>",
  message?: "<human string>" }` at the top level , flat, not the nested `{success:false,
  error:{code,message,details}}` shape the rest of the backend's `globalErrorHandler.ts` uses
  (`.claude/rules/api-routes.md`). **This is a genuinely different error shape from every other
  API in this codebase** , AGDK must parse `body.error` as the machine code and `body.message` as
  optional human text, never look for `body.error.code`.
- **`X-Matimo-Gateway-Overhead-Ms` response header**: set on every outcome of
  `/v1/chat/completions` and `/v1/messages` , success, streaming, and error alike (finding #8,
  2026-09-13, `gateway.ts:946`, `:1074`; `GatewayProxyService.ts:196-202` , the `error` variant of
  `GatewayChatCompletionOutcome` carries `overheadMs` specifically so this header is never skipped
  on a policy_denied/spend_cap_exceeded/etc. error). No other route sets this header.
- **No `GET /v1/health` or `GET /v1/models` route exists.** Confirmed by reading the entire file
  (`gateway.ts`, 1633 lines) , the only routes are the fourteen listed in §1.
- **Rate-limit response**: `429 { error: "rate_limit_exceeded" }`, no `Retry-After` header, no
  body detail (`gateway.ts:801`, `:812`, `:868`, `:779`). `GatewaySlidingWindowLimiter` fails
  **open** on a Redis error (`GatewaySlidingWindowLimiter.ts:18-21` doc comment) , a rate-limit
  check never itself 500s or 429s due to Redis being down.

---

## 1. Route table (method, path, scope, JWS, session)

| Method | Path | Scope required | Session required | JWS (`Matimo-Agent-Signature`) |
|---|---|---|---|---|
| POST | `/v1/chat/completions` | `gateway:proxy` | yes (`resolveSession`) | optional, enforced only if `identity.requireSignedRequests` OR `license.gatewayRequireSignedRequestsDefault` |
| POST | `/v1/messages` | `gateway:proxy` | yes | same opt-in enforcement as above |
| POST | `/v1/telemetry/batch` | `gateway:proxy` | yes | **not checked at all** (see Drift #2) |
| POST | `/v1/sessions` | `gateway:proxy` | n/a (this route mints the session) | **mandatory, unconditional** (`requireHandshakeSignature`) |
| DELETE | `/v1/sessions` | `gateway:proxy` | yes | no |
| POST | `/v1/identities` | `identity:manage` | no | no |
| POST | `/v1/identities/bulk` | `identity:manage` | no | no |
| GET | `/v1/identities/:agentId/jwks` | `identity:manage` | no | no |
| POST | `/v1/identities/:agentId/rotate-key` | `identity:manage` | no | no (admin action on the current key, doesn't need the old key to sign) |
| PUT | `/v1/routing-policies/:id/external-visibility` | `identity:manage` | no | no |
| POST | `/v1/tools/check` | `agdk:check` | no (`resolveIdentity`, bare token) | **mandatory, unconditional** (`requireAgdkSignature`) |
| POST | `/v1/tools/check/status` | `agdk:check` | no | mandatory |
| POST | `/v1/tools/result` | `agdk:check` | no | mandatory |
| PUT | `/v1/tools/:toolName/category` | `identity:manage` | no | no |

Source: every route definition in `gateway.ts:907-1631`. Every route also passes through
`apiKeyAuth` (`gateway.ts:301-328`) and `licenseGate` (`:330-360`) first, applied router-wide
(`router.use(apiKeyAuth); router.use(licenseGate);`, `:904-905`).

Middleware order actually enforced per route (matters for which error a bad request gets first):
1. `apiKeyAuth` → 2. `licenseGate` → 3. `requireScope(...)` → 4. `resolveSession` or
`resolveIdentity` (route-dependent) → 5. `verifyAgentSignature` / `requireHandshakeSignature` /
`requireAgdkSignature` (route-dependent) → 6. rate limiter → 7. handler's own Zod body parse.

---

## 2. Auth layer 1 , org API key

**Header**: `Authorization: Bearer <key>` (`extractBearerKey`, `gateway.ts:294-299`) , **not**
`Authorization: ApiKey <key>`, which is the separate, legacy `/api/v1/enterprise/*` scheme (§9).

**Key format**: `me-live-` + 32 lowercase-hex chars (`EnterpriseApiKeyService.ts:36`,
`randomBytes(24).toString('hex').substring(0,32)`). Hashed SHA-256 for storage/lookup
(`:99-101`). `key_prefix` stored is the first 14 chars (`'me-live-' + 6 hex chars`).

**Validation** (`EnterpriseApiKeyService.validate()`, `:55-79`): looks up by hash, rejects if
`!isActive` or `expiresAt < now`. On success returns `{id, tenantId, scopes, boundAgentId?}` and
fires a non-blocking `updateLastUsed()`. A bad/expired/unknown key → `401 { error:
"invalid_api_key" }` (`gateway.ts:316`).

**Scopes** (`EnterpriseApiKeyService.create()`, `:45`, default `['policy:check']` for keys created
through the *old* enterprise flow , a Gateway-facing key must be explicitly minted with
`['gateway:proxy', 'identity:manage', 'agdk:check']` or whichever subset it needs, confirmed
against `src/backend/scripts/gateway-manual-test.ts:683`, `:3230`). Scope check
(`requireScope`, `gateway.ts:362-376`): missing scope → `403 { error: "insufficient_scope",
message: "This API key is missing the required scope: <scope>" }`.

**License gate** (`licenseGate`, `gateway.ts:330-360`): fetches
`EnterpriseLicenseService.getLicense(tenantId)`. No license row, or `!isEnabled` → `403 {
error: "license_required", message: "An active Matimo Enterprise license is required for Matimo
Gateway" }`. Expired (`expiresAt < now`) → same code, message "...has expired". On success this
middleware also stashes `ctx.allowUnregisteredGatewayAccess` and `ctx.auditWriteMode` (`'sync'` if
`license.gatewaySyncAudit`, else `'async'`) for downstream use , **not** exposed to the client.

---

## 3. Agent identity registration

### 3.1. `POST /v1/identities` , primary front door (`gateway.ts:1280-1311`)

Auth: `Bearer <org API key>` with `identity:manage` scope. Rate limit: 500/hour per
`(tenantId, apiKeyId)` (`REGISTRATION_RATE_LIMIT`/`REGISTRATION_RATE_WINDOW_MS`,
`gateway.ts:129-130`; an anomaly-visibility log fires at 1000 attempts,
`REGISTRATION_ALERT_THRESHOLD`, non-blocking).

**Request body** (`registerIdentitySchema`, `gateway.ts:153-161`, `.strict()` , unknown fields
→ 400):
```
{
  displayName: string, min 1, max 255,
  externalFramework: "langchain" | "crewai" | "google-adk" | "custom",
  allowedToolCategories?: string[],
  allowedLlmModels?: string[],
  registrationMetadata?: Record<string, unknown>
}
```
`externalFramework` enum is `EXTERNAL_FRAMEWORKS` (`gateway.ts:147`), mirrored exactly by the DB
CHECK constraint `mai_external_framework_check` (`init.sql:4458`).

**Response**: `201 { data: { ...identity, privateKeyPem } }` (`gateway.ts:1309`). `identity` fields
come from `AgentIdentityService.registerExternalAgent()` → `rowToIdentity()`
(`AgentIdentityService.ts:41-71`):
```
{
  id, tenantId, novaAgentId: null, identityToken, displayName,
  riskClassification: "medium",   // always this literal for external registration (:239)
  lifecycleStatus: "active",
  allowedToolCategories: string[], allowedLlmModels: string[],
  externalFramework, publicKey: <SPKI PEM>, publicKeyFingerprint: <sha256 hex>,
  instructionVersion, parentIdentityId, maxSpawnDepth: 3,
  suspendedBy: null, revokedBy: null, registeredBy, autoClassified,
  createdAt, updatedAt, budgetCapEnabled: false, maxCostUsd: null, maxTokens: null,
  lastSeenAt: null, requireSignedRequests: false, lastTelemetryAt: null,
  privateKeyPem: "-----BEGIN PRIVATE KEY-----..."   // added only in this response
}
```

**Who generates the ECDSA keypair , the server does, not the client.**
`generateEcdsaKeyPair()` (`AgentIdentityService.ts:21-33`) calls
`generateKeyPairSync('ec', { namedCurve: 'P-256', publicKeyEncoding: {type:'spki',
format:'pem'}, privateKeyEncoding: {type:'pkcs8', format:'pem'} })` server-side. The **public**
key + its SHA-256 fingerprint are persisted to `matimo_agent_identities.public_key` /
`.public_key_fingerprint`. The **private** key (`privateKeyPem`, PKCS8 PEM) is returned in the
HTTP response body **exactly once** and is never stored anywhere server-side , `registerExternalAgent()`'s
return value is the only place it ever exists after generation (`AgentIdentityService.ts:227-275`).
The client must persist it locally; there is no retrieval endpoint. AGDK must save
`identityToken` and `privateKeyPem` together (see `agent.py:199-219`'s `persist_identity()`
pattern , separate files, `identityToken`/metadata in one, the PEM in another with `0600`
permissions on POSIX).

`identityToken` format: `me-id-` + 24 random alphanumeric chars from `[A-Za-z0-9]`
(`generateIdentityToken()`, `AgentIdentityService.ts:713-722`).

### 3.2. `POST /v1/identities/bulk` , M4 (`gateway.ts:1322-1354`)

Same auth/scope/rate-limit as 3.1 (one rate-limit check covers the whole batch, not per-item).

**Request**: `{ identities: RegisterIdentityBody[] }`, 1–50 items
(`bulkRegisterIdentitySchema`, `gateway.ts:170-174`, `.strict()`).

**Response**: `201 { data: [ ...per-item result ] }` where each item is either
`{ success: true, ...identity, privateKeyPem }` or `{ success: false, displayName, error:
<message> }` (`gateway.ts:1346-1352`). Never fails the whole batch on one bad row
(`AgentIdentityService.bulkRegisterExternalAgents()`, sequential execution, not `Promise.all`,
`AgentIdentityService.ts:337-375`).

### 3.3. `GET /v1/identities/:agentId/jwks` (`gateway.ts:1368-1393`)

Auth: `identity:manage` scope, no signature. `:agentId` must be a UUID (400 otherwise). 404
`{ error: "identity_not_found" }` if the identity doesn't exist for this tenant or has no
`publicKey`. Response: bare `{ keys: [<jwk>] }` , **not** `{data:...}**. One JWK
(`publicKeyPemToJwk`, `publicKeyJwk.ts:24-32`):
```
{ kty, crv, x, y, kid: <identityToken>, alg: "ES256", use: "sig" }
```
Not consumed by any current server verification path , verification reads
`matimo_agent_identities.public_key` directly (`gateway.ts:1359-1361` doc comment). Exists for
third-party independent verification later.

### 3.4. `POST /v1/identities/:agentId/rotate-key` , M4 (`gateway.ts:1406-1439`)

Auth: `identity:manage` scope, **no** JWS required (this is precisely the operation that must work
even if the caller lost the old private key). Generates a **fresh** ECDSA P-256 keypair,
overwrites `public_key`/`public_key_fingerprint` via `updatePublicKey()`, invalidates the 60s
identity Redis cache immediately (`AgentIdentityService.ts:300-321`). Does **not** touch
`identity_token` , the long-lived bearer token used to resolve the identity is untouched by key
rotation; only the *signing* key changes.

**Response**: `200 { data: { ...identity, privateKeyPem } }` (new one-time private key). 404
`{ error: "identity_not_found" }` if the identity doesn't exist (`gateway.ts:1436`).

**Client implication**: after rotation, the client must re-handshake (`POST /v1/sessions`) using
the *new* private key , any cached session token from before rotation is unaffected (sessions and
signing keys are orthogonal; see §4), but any subsequent *signed* call (handshake, `/tools/check`,
or a `requireSignedRequests`-enforced `/chat/completions` call) must sign with the new key or it
will fail verification.

### 3.5. Legacy front door: `POST /api/v1/enterprise/agents/register` (still live, unauthenticated by session)

Mounted at `/api/v1/enterprise` in `server.ts:621`, **before** the global auth gate , confirmed by
line-number comparison against the gate at `server.ts:819`. Auth: `Authorization: ApiKey <key>`
(note: `ApiKey`, not `Bearer` , different scheme from the `/v1` router entirely,
`external-policy-check.ts:40-46`). No scope check at all (D10, BUILD-PLAN §Part 6: "Existing
`/api/v1/enterprise/*` routes stay unscoped, a documented known gap").

Body: `{ displayName, externalFramework: string (min1,max100 , **not** the closed enum §3.1 uses),
allowedToolCategories?, allowedLlmModels?, registrationMetadata? }`
(`registerExternalSchema`, `external-policy-check.ts:26-34`). Response: `201 { data: {
...identity, privateKeyPem } }` , same underlying `AgentIdentityService.registerExternalAgent()`
call, `registeredBy: 'external_api'` (`:107-121`).

**AGDK should use `POST /v1/identities`, not this legacy route** , it is kept for backward
compatibility only and has a looser `externalFramework` validation and no scoped-key model.

### 3.6. Legacy status/heartbeat poll: `GET /api/v1/enterprise/agents/:token/status`

Same router/auth as 3.5 (`Authorization: ApiKey <key>`). Path param `:token` is the identity's
bearer `identityToken` (not a UUID). Response: `200 { data: { identityToken, displayName,
lifecycleStatus, riskClassification, externalFramework } }` (`external-policy-check.ts:161-169`).
404 `{ error: "Agent identity not found" }` if the token doesn't resolve for the validated
tenant , cross-tenant match is also 404 (never confirms existence under a different tenant,
`:156-159`).

**This is the only "poll my own status" endpoint that exists anywhere in the codebase** , see §7
for why it matters and what is *not* built.

---

## 4. Session handshake (mandatory before `/v1/chat/completions`, `/v1/messages`, `/v1/telemetry/batch`)

Added 2026-09-13, unconditional, no bypass, no transition period (`gateway.ts:46-64`). A bare
`X-Matimo-Agent-Identity-Token` bearer alone **no longer works** on these three routes.

### 4.1. `POST /v1/sessions` (`gateway.ts:1214-1252`)

**Auth chain**: `apiKeyAuth` → `licenseGate` → `requireScope('gateway:proxy')` → `resolveIdentity`
(resolves via `X-Matimo-Agent-Identity-Token` header, bare-token lookup,
`getIdentityByToken()`, uncached) → `sessionRateLimit` (30/hour per `(tenantId, identityId)`,
`SESSION_RATE_LIMIT`/`SESSION_RATE_WINDOW_MS`, `gateway.ts:113-114`) → `requireHandshakeSignature`
(**mandatory**, no opt-out , `gateway.ts:613-686`).

**Headers required**:
- `Authorization: Bearer <org API key>`
- `X-Matimo-Agent-Identity-Token: <identityToken>`
- `Matimo-Agent-Signature: <compact ES256 JWS>` , see §5 for the exact envelope. Signed over the
  **exact raw request body bytes actually sent**.

**Request body**: a literal empty JSON object, `{}`. `agent.py:283` sends the exact string `"{}"` as raw
bytes (via `content=`, deliberately not letting the HTTP client re-serialize a `json=` dict, which
could byte-differ from what was hashed into `body_hash`). The `body_hash` claim in the JWS must be
the SHA-256 hex of these exact bytes.

**Response**: `201 { data: { sessionToken, expiresAt, identityId } }`
(`gateway.ts:1250`). `sessionToken` format: `me-sess-` + 64 hex chars
(`randomBytes(32).toString('hex')`, `gateway.ts:1234`, `SESSION_TOKEN_PREFIX = 'me-sess-'`).
`expiresAt` is ISO 8601.

**TTL**: `min(license.gatewaySessionTtlSeconds ?? 3600, 86400)` , tenant-configurable via
`matimo_enterprise_licenses.gateway_session_ttl_seconds` (default `3600` = 1h,
`init.sql:4863`), hard-capped at 24h (`SESSION_TTL_CEILING_SECONDS`, `gateway.ts:109`) regardless
of license config. No admin UI to change this yet (read-only in practice as of this pass).

**Storage**: Redis only, key `matimo:gateway:session:<token>`, value
`JSON.stringify({identityId, tenantId})`, `SETEX` with the TTL above (`gateway.ts:1236-1240`).
**No database record of sessions exists** , the structured log line
`"[Matimo Gateway /v1] Session handshake succeeded"` plus Redis expiry is the entire audit trail.

**Renewal model , no refresh token.** There is deliberately no second, longer-lived
refresh-token secret. Renewal is always a fresh signed handshake using the same long-lived private
key (`gateway.ts:1207-1212` doc comment). The reference client renews proactively at 80% of the
session's own TTL (`SESSION_RENEWAL_FRACTION = 0.8`, `agent.py:160`, `:442-453`) and reactively on
a `401 session_expired` response (`agent.py:363-387`, `:455-465`).

### 4.2. `DELETE /v1/sessions` (`gateway.ts:1262-1278`)

Auth: `resolveSession` (i.e. you must already hold a valid, unexpired session token to delete it ,
you cannot delete an arbitrary session). Header: `X-Matimo-Session-Token`. Deletes the Redis key.
**Response: `204`, empty body.** Optional hygiene only , an unreleased session still expires
naturally via TTL; this is not load-bearing for correctness.

### 4.3. Using the session on `/v1/chat/completions`, `/v1/messages`, `/v1/telemetry/batch`

**Header**: `X-Matimo-Session-Token: <sessionToken>` , replaces
`X-Matimo-Agent-Identity-Token` on exactly these three routes (`resolveSession`,
`gateway.ts:448-518`). The long-lived identity token is **not** sent on these three calls once a
session exists.

**Resolution**: `redis.get(SESSION_KEY(token))` → parse `{identityId, tenantId}` → verify
`tenantId` matches the authenticated API key's tenant (cross-tenant treated identically to
unresolvable , never confirms existence under a different tenant, `gateway.ts:494-504`) →
`identityService.getIdentityById(identityId, tenantId)` (60s Redis-cached lookup, real-time
lifecycle re-checked downstream anyway) → if the identity was deleted since the session was
issued, `401 session_expired`.

**Every failure mode on this path collapses to one error**: `401 { error: "session_expired",
message?: <detail> }` , covers: missing header, unknown/expired Redis key, malformed cached JSON,
cross-tenant session, and identity-no-longer-exists (`gateway.ts:458-511`). This is deliberate
(mirrors `resolveIdentity`'s own "don't leak which part failed" posture) , **a client cannot
distinguish these five cases from the response alone; the only correct reaction to any
`session_expired` is to re-handshake.**

**Redis down**: fails **closed** here (unlike the rate limiter) , a Redis error resolving the
session returns `401 session_expired`, not a 500 (`gateway.ts:463-479`, explicit doc comment: "this
is the primary auth gate for these two routes post-handshake, not a rate-limit nicety").

**Revocation is real-time regardless of session TTL**: a suspend/revoke on the identity takes
effect on the very next call no matter how much of the session's TTL remains , lifecycle status is
re-checked fresh (not from the 60s cache) inside `AgentGatewayService.evaluate()` Step 4 on every
`/v1/chat/completions`/`/v1/messages` call (`AgentGatewayService.ts:574-579`,
`AgentIdentityService.getLifecycleStatus()` , "DB-direct, no cache" per that file's own comments).
A session token surviving does not mean the identity is still allowed to act.

---

## 5. Request signing (`Matimo-Agent-Signature` JWS)

Implementation: `verifyGatewayAgentSignature()` (`GatewaySignatureVerifier.ts`). Same envelope used
for the handshake (mandatory), `/v1/tools/*` (mandatory), and `/v1/chat/completions`/`/v1/messages`
(opt-in per identity/tenant).

### 5.1. JWS envelope

**Header**: `{ alg: "ES256", kid: "<identityToken>" }`. `alg` must be exactly `ES256`
(`GatewaySignatureVerifier.ts:47-49`). `kid` must equal the identity's own bearer `identityToken`
, checked as defense-in-depth, **not** the key-selection mechanism (the verification key always
comes from the identity already resolved server-side by `resolveIdentity`, never from the JWS
itself, `:12-18`).

**Claims** (all required unless noted):
```
{
  iss: "matimo-agdk",              // exact literal string, EXPECTED_ISSUER (:25)
  sub: "<identity.id>",            // the identity's UUID (matimo_agent_identities.id), NOT identityToken
  tenant_id: "<tenantId>",
  external_framework: "<identity.externalFramework>",  // required only if the identity has one set
  nonce: "<random string>",        // e.g. uuid4 hex, agent.py:264
  iat: <unix seconds>,
  exp: <unix seconds>,             // agent.py uses iat + 60
  body_hash: "<hex sha256>"        // sha256 of the raw request body bytes, hex-encoded, lowercase
}
```

**Signing algorithm**: ECDSA P-256 / ES256, signed with the identity's private key PEM
(PKCS8) exactly as issued at registration/rotation. Reference implementation uses PyJWT:
`pyjwt.encode(payload, private_key_pem, algorithm="ES256", headers={"kid": identityToken})`
(`agent.py:269-274`).

**`body_hash` computation**: `createHash('sha256').update(rawBody).digest('hex')`
(`GatewaySignatureVerifier.ts:98`) where `rawBody` is `req.rawBody` , the **exact bytes** Express's
`verify` callback captured (`gateway.ts:897-899`), i.e. exactly what was sent over the wire, no
re-serialization. **For `POST /v1/sessions`, the body is `{}` and the client must send that exact
2-byte string** (not `{ }` with a space, not a re-`JSON.stringify()`'d empty object from a
different library that might format differently) , `agent.py:283`/`:293` sends raw bytes via
`content=` specifically to guarantee this. For `/v1/chat/completions`/`/v1/messages`/`/v1/tools/*`,
hash the exact JSON bytes the HTTP client library actually transmits as the request body.

**Clock skew / replay window**: `±60 seconds` on `iat` relative to the server's clock
(`CLOCK_SKEW_WINDOW_SECONDS = 60`, `GatewaySignatureVerifier.ts:28`, `:86`). `exp` must not be in
the past relative to server time (`:89-91`) , reference client sets `exp = iat + 60`.

**Nonce / replay protection**: `GatewaySignatureNonceStore.checkAndConsume(identityToken, nonce)`
, Redis `SET key '1' EX 120 NX` (`GatewaySignatureNonceStore.ts:36-40`), keyed
`matimo:gateway:sig-nonce:<identityToken>:<nonce>`. TTL 120s (double the ±60s window). **A reused
nonce for the same identity is rejected** , generate a fresh random nonce per signed request,
never reuse one.

**Verification failure reasons** (`GatewaySignatureVerificationResult`, all map to the same `403
signature_required` when enforcement is on): missing/wrong `alg`, `kid` mismatch, signature
verification failure (wrong key / tampered payload), wrong `iss`, `sub` mismatch, `tenant_id`
mismatch, `external_framework` mismatch, `iat` outside ±60s, expired, missing `nonce`,
`body_hash` mismatch, or replayed nonce (`GatewaySignatureVerifier.ts:41-108`). The response never
distinguishes which , always `403 { error: "signature_required", message: "Matimo-Agent-Signature
is missing or invalid" }`.

### 5.2. Enforcement posture per route

| Route(s) | Enforcement |
|---|---|
| `POST /v1/sessions` | **Always mandatory**, no opt-out (`requireHandshakeSignature`). No signature at all → `403 signature_required`. |
| `POST /v1/tools/check`, `/check/status`, `/result` | **Always mandatory** (`requireAgdkSignature`). |
| `POST /v1/chat/completions`, `POST /v1/messages` | **Opt-in**: enforced only if `identity.requireSignedRequests === true` OR `license.gatewayRequireSignedRequestsDefault === true` (an OR , either flag being on requires it for that identity's calls). If enforcement is **off** and the header is absent or invalid, the call still proceeds , a failure is logged but never rejected (`verifyAgentSignature`, `gateway.ts:520-611`). Both flags default `false` (`init.sql:4451`, `:4844`) and have no admin UI to change yet , read-only in practice. |
| Everything else (`/v1/identities*`, jwks, rotate-key, routing-policy visibility, tool category) | No signature at all. |

**AGDK implication**: sign every `/v1/chat/completions`/`/v1/messages` call once the client holds
a private key anyway (registration already required one) , there is no reason not to, since the
server only rejects on a *bad* signature when enforcement is on, and silently accepts a *present
but currently-unenforced* signature the rest of the time (useful for a tenant testing their
rollout before flipping `requireSignedRequests` on).

---

## 6. LLM proxy , `POST /v1/chat/completions` and `POST /v1/messages`

### 6.1. Model naming / routing (`GatewayRoutingService.selectTarget()`, `GatewayRoutingService.ts:53-61`)

Three modes, decided by the wire `model` field:
- **Omitted** (`model` absent/undefined): uses the tenant's default-fallback BYOK connection
  (`ILLMProviderRepository.findDefaultFallbackByTenant()`), first model in that connection's
  `models` array. Errors `400 no_default_connection` if none configured.
- **`"matimo/auto"`** (exact sentinel string, `AUTO_SENTINEL`, `GatewayRoutingService.ts:17`):
  routes through the tenant's active `RoutingPolicy` marked `exposedToGateway: true` (see
  `PUT /v1/routing-policies/:id/external-visibility` to expose one) , classifies task complexity
  from the first user message via the same rule-based classifier Nova's own `SmartRoutingService`
  uses, then picks the policy's model for that complexity tier. Errors `400
  no_gateway_routing_policy` if zero policies are exposed.
- **Any other string** (a pin, e.g. `"gpt-4o"`, `"claude-3-5-sonnet-20241022"`): must be a
  literal model name present in one of the tenant's active, `credential_source: 'byok'`
  `llm_providers.models` arrays, **and** (if the identity's `allowedLlmModels` is non-empty) must
  be in that list too. Errors `403 model_not_allowed` if either check fails.

`allowedLlmModels: []` (empty array) means unrestricted (same convention as
`allowedToolCategories`).

**Gateway is BYOK-only in v1** , a resolved target naming a `nova_managed`/`nova_credits`
connection throws `GatewayNonByokConnectionError` → `400 { error: "non_byok_connection" }`
(`GatewayRoutingService.ts` / `types.ts:56-66`).

### 6.2. `POST /v1/chat/completions` , OpenAI-compatible wire

**Request schema** (`openAIChatCompletionRequestSchema`, `openaiWireFormat.ts:82-99`, **not**
`.strict()` , unknown OpenAI fields like `n`/`presence_penalty`/`user` pass through harmlessly and
are simply dropped, never cause a 400):
```
{
  model?: string,
  messages: [{ role: "system"|"user"|"assistant"|"tool", content: string,
               tool_call_id?: string,   // required if role === "tool"
               tool_calls?: [{ id, type: "function", function: { name, arguments: string } }] }],
              // min 1 message
  temperature?: number, max_tokens?: number (int, positive), top_p?: number,
  stop?: string | string[], stream?: boolean,
  tools?: [{ type: "function", function: { name, description?, parameters: {
              type:"object", properties?, required?[] } } }],
  tool_choice?: string | { type: "function", function: { name } }
}
```
`content` is a plain string only , **no multimodal content-part arrays**. A request using that
shape is rejected `400 invalid_request` at the Zod boundary, never silently mangled
(`openaiWireFormat.ts:1-9`).

**Tool-calling round-trip**: fully supported when the resolved backend provider is OpenAI
(`role: 'tool'` messages with `tool_call_id`, and `tool_calls` on the preceding assistant message,
both required to reconstruct a real multi-turn ReAct loop, per BUILD-PLAN finding F22). **Known
gap, not fixed**: if Gateway's router resolves to a non-OpenAI backend (Anthropic/Google/Bedrock),
that provider's own pre-existing translation layer doesn't yet turn a `role:'tool'` `LLMMessage`
into its native tool-result format , tracked as F22/Q8, not silently assumed to work
(`openaiWireFormat.ts:26-31`).

**Response** (non-streaming, `200`): standard OpenAI chat-completion shape
(`OpenAIChatCompletionResponse`, `openaiWireFormat.ts:163-182`):
```
{ id: "chatcmpl-<uuid>", object: "chat.completion", created: <unix s>, model: <resolved model>,
  choices: [{ index: 0, message: { role: "assistant", content: string|null, tool_calls?: [...] },
              finish_reason: "stop"|"length"|"content_filter"|"tool_calls" }],
  usage: { prompt_tokens, completion_tokens, total_tokens } }
```

**Streaming** (`stream: true`): standard OpenAI SSE, `Content-Type: text/event-stream`, each
frame `data: <json>\n\n`, terminal `data: [DONE]\n\n` (`gateway.ts:1006-1013`). Headers
`Cache-Control: no-cache, no-transform`, `Connection: keep-alive`, flushed before the first event.

### 6.3. `POST /v1/messages` , Anthropic-compatible wire (M4)

**Request schema** (`anthropicMessagesRequestSchema`, `anthropicWireFormat.ts:90-108`, not
`.strict()`):
```
{
  model?: string,
  system?: string | [{ type:"text", text }],
  messages: [{ role: "user"|"assistant",
               content: string | [ {type:"text",text} | {type:"tool_use",id,name,input}
                                    | {type:"tool_result", tool_use_id, content?, is_error?} ] }],
  max_tokens: number (int, positive) , REQUIRED, matches the real Anthropic API contract,
  temperature?, top_p?, top_k?, stop_sequences?: string[], stream?: boolean,
  tools?: [{ name, description?, input_schema: {type:"object", properties?, required?[]} }],
  tool_choice?: {type:"auto"} | {type:"any"} | {type:"tool", name}
}
```
Same text-only content restriction as the OpenAI shim , no image/document content blocks.
`{type:"any"}` tool_choice has no `LLMConfig` equivalent and degrades to provider-default `auto`
(disclosed scope-down, `anthropicWireFormat.ts:203-206`).

**Response** (non-streaming, `200`):
```
{ id: "msg_<uuid>", type:"message", role:"assistant", model,
  content: [{type:"text",text}, ...{type:"tool_use",id,name,input}...],
  stop_reason: "end_turn"|"max_tokens"|"stop_sequence"|"tool_use"|"content_filter",
  stop_sequence: null,
  usage: { input_tokens, output_tokens } }
```
**Drift flagged in the code itself**: `"content_filter"` is used for a guardrail-substituted
response per TRD §3b, but Anthropic's real public `stop_reason` enum
(`end_turn|max_tokens|stop_sequence|tool_use|pause_turn|refusal`) has no `content_filter` value ,
`anthropicWireFormat.ts:222-235` flags this explicitly as unresolved daylight between the design
doc and the real API, not something this code silently invented without disclosure. It does not
break an unmodified Anthropic SDK client (the field is an untyped string), but AGDK should not
assume every real Claude SDK treats it meaningfully.

**Streaming**: real Anthropic multi-event envelope, no `[DONE]` marker , connection simply closes
after `message_stop`:
```
event: message_start\ndata: {...}\n\n
event: content_block_start\ndata: {...}\n\n
event: content_block_delta\ndata: {...}\n\n   (repeated)
event: content_block_stop\ndata: {...}\n\n
event: message_delta\ndata: {...}\n\n
event: message_stop\ndata: {...}\n\n
```
(`anthropicWireFormat.ts:307-383`).

### 6.4. Governed pipeline order (both routes, shared verbatim via `prepareRequest()`/`dispatchAndFinish()`)

1. Guardrails `evaluateInput()` on the current turn only (last user message + any trailing
   `role:'tool'` messages after it , never the full history).
2. `AgentGatewayService.checkExternalAgent()` , license, emergency stop, identity resolution,
   lifecycle, telemetry-staleness (external agents only), monthly budget, spawn depth, access
   grants, Policy Engine.
3. `GatewayRoutingService.selectTarget()`.
4. `GatewaySpendCapService.assertWithinCap()` (before dispatch).
5. Agent Soul injection into the outgoing system message.
6. Provider dispatch (with pre-first-byte fallback chain).
7. Guardrails `evaluateOutput()` on the response content + tool-call args (or the streaming
   equivalent).
8. Wire-translate the response.
9. Fire-and-forget (or, if `gateway_sync_audit`, awaited) audit write.

(`GatewayProxyService.ts:1-66` class doc comment; verified against the actual method bodies.)

### 6.5. Guardrail outcomes are NOT HTTP errors

**Critical for client design**: an input or output guardrail `block`, `redirect`, or
`hold_for_review` outcome on `/v1/chat/completions`/`/v1/messages` does **not** produce a non-200
response. It produces a normal `200` (or a normal SSE stream) whose content is the guardrail's
substituted/redirect message, with `finish_reason` / `stop_reason` set to `"content_filter"`
(`GatewayProxyService.ts:384-402`, `isSubstitutingOutcome()`, `:336-338`). **`hold_for_review` at
this layer does not pause/wait for a human decision** , unlike Nova's own `AgentExecutionService`
path, Gateway's proxy substitutes the message immediately and marks the run
`awaiting_guardrail_review` for observability only; there is nothing to poll. A client should treat
`finish_reason === "content_filter"` as "this response was intercepted by a guardrail," not as an
error to retry.

### 6.6. Error responses (all as `{ error: "<code>", message?: string }`)

| Status | code | When | message present? |
|---|---|---|---|
| 400 | `invalid_request` | Zod schema failure on the request body | yes, Zod's message |
| 403 | `model_not_allowed` | pinned model not in tenant's BYOK connections or identity's `allowedLlmModels` | yes |
| 400 | `no_gateway_routing_policy` | `matimo/auto` requested, no policy exposed | yes |
| 400 | `no_default_connection` | model omitted, no default BYOK connection | yes |
| 400 | `non_byok_connection` | resolved target is `nova_managed`/`nova_credits` | yes |
| 403 | `policy_denied` | `AgentGatewayService` returned DENY or `emergency_stop` | **yes , `message` is the machine-readable `decision.reason`** (see table below), not generic text |
| 403 | `spend_cap_exceeded` | identity's per-call budget cap tripped | **no message field at all** |
| 502 | `upstream_error` | all routing targets (primary + fallbacks) failed to dispatch | yes, fixed string "All configured LLM targets failed" |

**`policy_denied`'s `message` is the actual denial reason string, not decorative text** , a client
that wants to distinguish *why* it was denied must parse `message`, since `error` is always the
same literal `"policy_denied"`. Observed reason strings (`AgentGatewayService.ts`):
`agent_identity_not_found`, `agent_suspended`, `agent_revoked`, `emergency_stop_active`,
`telemetry_stale` (only when `license.telemetryMode === 'deny'`, see §7.2),
`monthly_budget_exceeded`, `spawn_depth_exceeded`, `tool_category_not_allowed` (tool-check path
only), `invalid_api_key`, `api_key_not_bound_to_agent`, `license_required`,
`gateway_error_fail_closed`, `gateway_infra_error_fail_closed`, or a policy-engine-authored string
(`Policy "<name>" denied this tool call` or a custom `reason` on the matched rule).

**Rate limit**: `429 { error: "rate_limit_exceeded" }` (tenant+key: 6000/hour; per-identity:
1200/hour, `gateway.ts:122-125`, `:786-818`).

**Auth/scope/session/signature errors** (apply to every route, listed once): `401
invalid_api_key`, `403 insufficient_scope`, `403 identity_required` (bare-token routes only), `401
session_expired` (session routes only), `403 signature_required` (signed routes only), `403
license_required`.

---

## 7. Telemetry

### 7.1. `POST /v1/telemetry/batch` (`gateway.ts:1154-1190`)

Auth: `gateway:proxy` scope + `resolveSession` (mandatory session, same as the LLM proxy) +
`proxyRateLimit` (reuses the LLM proxy's own rate limiter, no dedicated telemetry limiter).
**No JWS signature check on this route at all** , see Drift #2.

Two body shapes accepted, tried in order , the bespoke envelope first, falling back to OTLP only
if that fails to parse:

**Shape A , AGDK's bespoke envelope** (`telemetryBatchSchema`, `gateway.ts:205-209`, `.strict()`
at the top level; the per-event schema is *not* strict, so unknown fields on an event pass through
harmlessly):
```
{ events: [ {
    runId: string, min1 max120,             // REQUIRED
    sessionId?: string max120,
    spanId?: string max120,
    parentSpanId?: string max120,
    kind: "run" | "llm" | "tool" | "log" | "error",   // REQUIRED
    name?: string max255,
    status?: string max20,
    startedAt?: string (ISO 8601) | number (epoch ms),
    durationMs?: number (int, non-negative),
    attributes?: Record<string, unknown>
} ], // 1 to 500 events per batch
}
```

**Shape B , raw OTLP/HTTP JSON `ExportTraceServiceRequest`** (`otlpExportTraceServiceRequestSchema`,
`otlpTelemetryAdapter.ts:109-111`): `{ resourceSpans: [{ resource?, scopeSpans: [{ scope?, spans:
[...] }] }] }`, standard OTLP trace shape, loose/passthrough on unknown fields. `traceId` becomes
`runId` unless a `matimo.run_id` resource/span attribute overrides it. `gen_ai.operation.name`
picks the row's `kind` (`invoke_agent|invoke_workflow|create_agent`→`run`,
`chat|text_completion|generate_content|embeddings`→`llm`, `execute_tool`→`tool`, an OTel error
status with no recognized operation→`error`, else→`log`). `TraceId`/`SpanId` accepted as hex or
base64. **AGDK itself should almost certainly emit Shape A** (it's the native, simpler contract);
Shape B exists for third-party OTel exporters a tenant already has, not as AGDK's primary path.

**Response**: `200 { data: { accepted: number, failed: [{ index, error }] } }`
(`GatewayTelemetryIngestService.ts:57-60`). Per-event, not all-or-nothing , one bad event doesn't
lose the rest of the batch.

**Masking**: every event's `attributes`, after `gen_ai.*` gap-filling, is deep-masked
(PII via `redactPiiDeep()`, then secrets via a global-flag clone of `SECRET_PATTERNS`) **before**
storage , recursively, including nested objects/arrays (`telemetryMasking.ts`). A field the caller
sends will come back (if ever read back) redacted as `[REDACTED:<CLASS>]` if it matches a known PII
or secret pattern; anything else passes through untouched. Do not rely on sending anything
sensitive expecting it to survive intact.

**`gen_ai.*` attribute conventions** (`telemetryAttributeShape.ts`): the server fills gaps, never
overwrites what the caller sent. It derives `gen_ai.operation.name` from `kind`
(`run`→`invoke_agent`, `llm`→`chat`, `tool`→`execute_tool`) only if the caller didn't already set
it, and `gen_ai.tool.name` from `name` for `kind:'tool'` events only if absent. **AGDK should send
its own, more specific `gen_ai.operation.name`** (e.g. `generate_content` for Gemini) whenever it
knows a better value than the coarse per-`kind` default.

### 7.2. Telemetry staleness → next-call deny (`AgentGatewayService.evaluate()` Step 4.5)

Opt-in per tenant via `matimo_enterprise_licenses.telemetry_mode` (`'advisory'` default | `'deny'`)
and `.telemetry_staleness_minutes` (default `30`). Only applies to **external agents**, on **every**
`/v1/chat/completions`/`/v1/messages` call, not just the first stale one
(`AgentGatewayService.ts:605-629`):
- If `telemetryMode !== 'deny'`: never denies (advisory only , no current UI surfaces the
  advisory signal either, per TELEMETRY.md §8).
- If `telemetryMode === 'deny'` and `identity.lastTelemetryAt === null` (this identity has never
  once sent telemetry): **bootstrap grace , never denied** for having not-yet-reported telemetry
  on its very first call.
- If `telemetryMode === 'deny'` and `Date.now() - lastTelemetryAt > telemetryStalenessMinutes *
  60000`: `403 { error: "policy_denied", message: "telemetry_stale" }`.

**Heartbeat cadence recommendation** (not itself specified anywhere in code, derived from the
above): with the default 30-minute staleness window and a 5-minute sweep tick
(`GovernanceApprovalExpiryService`/`GatewayRunStalenessSweepService` both tick every 5 min), a
client should push at least one telemetry event well under 30 minutes , e.g. every 5–10 minutes,
or before every LLM call if the agent calls more frequently than that , to avoid both (a) the
identity being denied under `deny` mode and (b) its `matimo_gateway_runs` row going `stale`
(`matimo_gateway_runs.status` flips to `'stale'` after `last_activity_at` exceeds the same
`telemetry_staleness_minutes` window, `GatewayRunStalenessSweepService.ts:14-16`, 5-min tick).
**Note the same config value drives two independent mechanisms** , run staleness (cosmetic, just a
status label) and identity deny (a hard 403) , don't confuse "my run shows stale" with "my calls
are being denied"; only the latter is `telemetryMode === 'deny'`-gated.

**`X-Matimo-Run-Id` header** (optional, `gateway.ts:116-117`, `:285-291`): correlates a
`/v1/chat/completions`/`/v1/messages` call's audit row, guardrail evaluations, and
`matimo_gateway_runs` rolling summary with telemetry events sharing the same `runId`. Max 120
chars, silently truncated (never 400s) if longer. A bare `base_url`-swap caller with no AGDK never
sends this , Gateway works identically either way. **`sessionId` on a telemetry event should equal
`runId`** if you want `matimo_gateway_runs.session_id` populated (found live-testing: omitting it
means the run's "session: ..." badge never appears in the admin UI, `agent.py:40-46`).

### 7.3. Run status lifecycle (for context , not a client concern beyond §7.2)

`matimo_gateway_runs.status`: `running → {completed|failed|cancelled}` (only via an explicit
`kind:'run'` telemetry span whose `status` is `completed|success|ok` / `failed|error` /
`cancelled`), or `running → blocked_by_guardrail` / `awaiting_guardrail_review` (via a guardrail
outcome on a call in that run), or `running → stale` (sweep, see above). **A bare successful LLM
call never itself marks a run `completed`** , only an explicit terminal telemetry span or the
staleness sweep ends a run (`TELEMETRY.md` §3).

---

## 8. Tool governance (M3) , `/v1/tools/*`

All three POST routes use `resolveIdentity` (bare `X-Matimo-Agent-Identity-Token`, **not**
session-based) + mandatory `requireAgdkSignature` (§5). Scope `agdk:check`.

### 8.1. `POST /v1/tools/check` (`gateway.ts:1490-1522`)

**Request** (`toolCheckSchema`, `gateway.ts:220-231`, `.strict()`):
```
{
  toolName: string, min1 max255,
  toolCategory?: string max50,   // a HINT only , server always re-resolves the real category, never trusted for the policy decision itself
  argHash: string, min1 max128,  // the caller's own hash of the tool call's args , Gateway never sees raw args for the actual policy check
  args?: Record<string, unknown> // OPTIONAL , only for a human reviewer's benefit if outcome=PENDING; masked (PII/secrets) before ever persisted
}
```

**Idempotency**: the dedup key is **server-derived**, `sha256(identityToken:toolName:argHash)`
(`GatewayToolCheckService.ts:70-72`) , never trust/accept a client-supplied idempotency key. Cached
in Redis for **15 minutes** (`IDEMPOTENCY_TTL_SECONDS`, `:32`) , an identical retry within that
window gets back the exact same decision, including the same `resumeToken` if `PENDING`, not a new
approval row.

**Response**: `200 { data: { decision: "ALLOW"|"DENY"|"PENDING", reason?, resumeToken?,
requestId? } }` (`ToolCheckResponse`, `GatewayToolCheckService.ts:46-51`). `resumeToken` and
`requestId` are set **only** when `decision === "PENDING"`. A concurrent duplicate-in-flight race
(same dedup key, two callers) returns `{ decision: "PENDING", reason: "duplicate_check_in_flight"
}` with **no resume token** , the caller should poll `/tools/check/status` shortly after, using
whatever resume token the *other* caller of that identical check eventually surfaces (or simply
re-issue `/tools/check` again after a short delay , it will return the cached PENDING decision with
its real resume token once that request has actually landed).

**Outcome semantics** (`AgentGatewayService.checkToolCall()`, `AgentGatewayService.ts:289-430`):
- `DENY` reasons: `license_required`, `emergency_stop_active`, `agent_identity_not_found`,
  `agent_suspended`, `agent_revoked`, `tool_category_not_allowed` (identity's
  `allowedToolCategories` non-empty and doesn't include this tool's category), or a matched
  Policy Engine `DENY` rule's reason.
- `PENDING`: Policy Engine matched a rule with outcome `ALLOW_WITH_CONDITIONS` , creates a
  `matimo_enterprise_governance_approvals` row (`request_kind: 'tool_check'`), TTL **4 hours**
  (`TOOL_CHECK_DEFAULT_TTL_HOURS`, `GovernanceApprovalService.ts:58`), and returns a fresh
  `resumeToken` (64 hex chars, `randomBytes(32).toString('hex')`) , only its SHA-256 hash is
  persisted server-side; the plaintext is returned exactly once and cannot be recovered.
- `ALLOW`: everything else.

**Fail behavior on an internal exception** (`GatewayToolCheckService.failSafeDecision()`,
`:159-178`): maps to the tenant's `gateway_fail_behavior` (`FAIL_CLOSED` default). `FAIL_OPEN` →
`{decision:"ALLOW", reason:"gateway_infra_error_fail_open"}`. Everything else (including
`FAIL_OPEN_BOUNDED`, which has **no** bounded-cache replay for tool checks , it degrades straight to
`FAIL_CLOSED`, a documented simplification) → `{decision:"DENY",
reason:"gateway_infra_error_fail_closed"}`. Never a raw 500.

### 8.2. `POST /v1/tools/check/status` (`gateway.ts:1529-1557`)

**Request**: `{ resumeToken: string, min1 }` (`toolCheckStatusSchema`, `.strict()`). **The resume
token travels in the body, never the URL** , deliberate, to avoid it landing in access/proxy logs
or browser history.

**Response**: `200 { data: { decision: "ALLOW"|"DENY"|"PENDING", reason? } }`. `404 {
error: "resume_token_not_found" }` if the token doesn't resolve for this tenant (never
distinguishes wrong-tenant from doesn't-exist).

**Polling contract** (`GovernanceApprovalService.pollToolCheck()`, `:269-287`):
- Still `pending` and `expires_at > now` → `{decision: "PENDING"}` , keep polling.
- `approved` → `{decision: "ALLOW"}`.
- `rejected` → `{decision: "DENY", reason: <decision_reason or "rejected">}`.
- `expired` (either swept, or still DB-status `pending` but past its own `expires_at` before the
  sweep catches it) → `{decision: "DENY", reason: "expired"}` , **fail-closed default for an
  unresolved HITL gate**.

**Expiry sweep**: `GovernanceApprovalExpiryService`, ticks every **5 minutes**
(`TICK_INTERVAL_MS`, `GovernanceApprovalExpiryService.ts:26`), bulk `UPDATE ... WHERE
status='pending' AND expires_at <= NOW()`. **No specific recommended client poll interval is
specified anywhere in code or docs** , given the 4-hour default TTL and 5-minute sweep granularity,
a poll interval in the 5–30 second range is reasonable for interactive use; nothing server-side
rate-limits this route beyond the shared `proxyRateLimit` (1200/hour per identity).

### 8.3. `POST /v1/tools/result` (`gateway.ts:1564-1592`)

**Request**: `{ resumeToken: string min1, status: string min1 max20, durationMs?: number (int,
non-negative), error?: string max2000 }` (`toolResultSchema`, `.strict()`).

**Response**: `202 { data: { accepted: true } }` , **always**, unconditionally. This is
**best-effort, log-only** in the current implementation (`GatewayToolCheckService.recordResult()`,
`:187-199` , just calls `logger.info()`, nothing is persisted to any queryable table, not wired
into telemetry). Optional; a client can skip calling this entirely with no functional
consequence today.

### 8.4. `PUT /v1/tools/:toolName/category` (`gateway.ts:1600-1631`)

Auth: `identity:manage` scope (**not** identity-scoped , this is a tenant-wide admin action, no
`resolveIdentity`/signature). Request: `{ category: string, min1 max50 }`
(`toolCategorySchema`, `.strict()`). Response: `200 { data: <updated row> }`. This is how an admin
(or an AGDK bootstrap script run with an admin-scoped key) pre-classifies a tool's category ahead
of `/tools/check` ever needing to guess one.

---

## 9. Scopes summary

| Scope | Grants |
|---|---|
| `gateway:proxy` | `/v1/chat/completions`, `/v1/messages`, `/v1/telemetry/batch`, `/v1/sessions` (both POST and DELETE) |
| `identity:manage` | `/v1/identities`, `/v1/identities/bulk`, `/v1/identities/:id/jwks`, `/v1/identities/:id/rotate-key`, `/v1/routing-policies/:id/external-visibility`, `/v1/tools/:toolName/category` |
| `agdk:check` | `/v1/tools/check`, `/v1/tools/check/status`, `/v1/tools/result` |

A single AGDK-facing API key should typically carry all three (confirmed pattern:
`gateway-manual-test.ts:3230` mints `['gateway:proxy', 'identity:manage', 'agdk:check']` together
for its full-flow scenarios). Missing any one scope → `403 insufficient_scope` naming the missing
scope in `message`.

---

## 10. What does NOT exist (don't build a client assuming these)

- **No `GET /v1/models` endpoint.** A client cannot ask Gateway which models are available; it
  must know the pin ahead of time, use `matimo/auto`, or omit `model` to get the tenant's default.
- **No `GET /v1/health` endpoint** under this router.
- **No push-based kill switch / rapid-suspend.** `TELEMETRY.md` §8 states this explicitly: "nothing
  in this pipeline currently halts a *running* agent loop." The only mechanism that ever stops a
  misbehaving agent is the **next** `/v1/chat/completions`/`/v1/messages`/`/v1/tools/check` call
  being denied (lifecycle check re-reads the DB directly, uncached, every single call , so
  suspend/revoke/emergency-stop take effect immediately on the *next* call, not on the current
  one). A long-running single call already in flight is not interrupted.
- **No bypass-detection** (an identity whose telemetry keeps flowing while its LLM traffic
  silently stops going through Gateway, or vice versa) , confirmed zero code for it.
- **No AGDK-specific heartbeat/poll endpoint distinct from the legacy `GET
  /api/v1/enterprise/agents/:token/status`** (§3.6). If AGDK wants a "did I get suspended" signal
  independent of making an actual governed call, that legacy route (different auth scheme ,
  `Authorization: ApiKey`, not `Bearer`) is the only thing that exists today. No recommended poll
  interval is specified anywhere for it.
- **`POST /v1/tools/result` persists nothing queryable** , treat it as fire-and-forget telemetry,
  not an audit mechanism a compliance report could rely on today.

---

## 11. Client design implications (what AGDK MUST get right)

1. **Registration is one-time per machine/identity, not per run.** Persist `identityToken` +
   `privateKeyPem` to local storage after `POST /v1/identities`; there is no way to retrieve the
   private key again. Treat losing it as needing a fresh registration (or a key rotation, if the
   identity + org API key are both still known).
2. **A session handshake is mandatory before the first `/v1/chat/completions`, `/v1/messages`, or
   `/v1/telemetry/batch` call of a process's lifetime** , do this immediately after resolving/
   registering the identity, before building any provider client. There is no fallback path.
3. **Sign the handshake body exactly as `{}`, as raw bytes, and hash those same raw bytes into
   `body_hash`.** Never let a JSON library independently re-serialize the body between hashing and
   sending , a byte mismatch (e.g. whitespace difference) fails `body_hash` verification with no
   distinguishing error message.
4. **Renew sessions proactively** (e.g. at 75–85% of the returned TTL) **and reactively** on any
   `401` whose body is `{error: "session_expired"}` , check `error === "session_expired"`
   specifically, not "any 401," since `invalid_api_key`/`license_required`/etc. also return 401/403
   and are not recoverable by re-handshaking.
5. **Generate a fresh random nonce per signed request** (handshake, `/tools/check` family, and any
   signed `/chat/completions` call) , a reused nonce for the same identity within 120 seconds is
   rejected as a replay.
6. **Sign `/v1/chat/completions`/`/v1/messages` calls whenever a private key is available, even if
   the tenant hasn't enforced `requireSignedRequests` yet** , it costs nothing when unenforced and
   is required the moment a tenant flips enforcement on; retrofitting signing later is strictly
   harder than always doing it.
7. **Treat `finish_reason`/`stop_reason: "content_filter"` as a governed-block signal, not an
   error** , it arrives inside a normal 200/stream response, not as an HTTP error. Do not retry it
   as if it were a transient failure.
8. **Retry/backoff**: `429 rate_limit_exceeded` and `502 upstream_error` are the only responses that
   plausibly warrant a retry with backoff. Every `4xx` other than `429` reflects a real
   configuration/policy/auth problem that a retry will not fix (in particular, `403 policy_denied`
   will simply recur , check `message` for the reason and surface it, don't loop on it).
9. **On `telemetry_stale` denials specifically**: this only happens under `telemetryMode='deny'`
   after the identity has sent telemetry at least once and then gone quiet past the staleness
   window. The fix is resuming telemetry pushes, not retrying the LLM call , retrying without
   sending telemetry first will deny again identically.
10. **`X-Matimo-Run-Id` and telemetry's `sessionId` should carry the same value per logical run**,
    and that value should be sent on every LLM call and every telemetry event belonging to that run
    , omitting `sessionId` on telemetry silently drops the run's session correlation in the admin
    UI, a real regression the reference client hit and fixed.
11. **Never construct the tool-check idempotency/dedup key client-side** , it is entirely
    server-derived from `(identityToken, toolName, argHash)`. The client only needs to compute
    `argHash` (a hash of its own choosing, over the tool's actual arguments) consistently for the
    same logical call.
12. **A `PENDING` tool-check needs real polling** (`/v1/tools/check/status`, resume token in the
    body) , unlike a guardrail hold on the chat-completions path, this one genuinely blocks until a
    human decides or the request expires (default 4h). Poll on a short interval (seconds, not
    minutes) if the calling agent is interactive; there is no server-provided poll-interval hint.
13. **Bulk-register (`/v1/identities/bulk`) returns partial success** , always check each item's
    own `success` field; a 201 status does not mean every identity in the batch was created.
14. **After a key rotation, re-handshake with the new key immediately** , an old session token
    obtained before rotation is not itself invalidated by rotation (sessions and signing keys are
    independent), but any new signed operation must use the new key.
15. **Don't rely on `POST /v1/tools/result`'s response for anything** , it's a fire-and-forget log
    line server-side today, not a persisted audit record.

---

## Drift , where the design docs and the live code disagree

1. **BUILD-PLAN Part 6's contract table is stale relative to the current router.** It lists only
   ten routes and predates `POST /v1/sessions`, `DELETE /v1/sessions` (added 2026-09-13, mandatory
   handshake), and `POST /v1/messages`/bulk-registration/key-rotation (M4, 2026-09-16). Trust the
   live route list in §1 above, not that table, for what currently exists.
2. **BUILD-PLAN Part 6's table says `POST /v1/telemetry/batch` requires JWS ("yes")** , the live
   code does **not** apply any signature-verification middleware to this route at all
   (`gateway.ts:1154-1158`: only `requireScope`, `resolveSession`, `proxyRateLimit` , no
   `verifyAgentSignature`/`requireAgdkSignature`). The route's own doc comment
   (`gateway.ts:1132-1137`) confirms this is deliberate: "extending [D18's JWS] to telemetry
   ingestion is an M2b decision... not implied by this one." The design doc's contract table simply
   was never updated to reflect that telemetry signing was descoped. **AGDK must not sign telemetry
   batch calls expecting the server to check it , it doesn't, on this route.**
3. **D14 (BUILD-PLAN) describes the telemetry response as carrying "the heartbeat"
   (lifecycle/emergency-stop/config-version).** The live response is just `{ accepted, failed }}`
   (`GatewayTelemetryIngestService.ts:57-60`) , no lifecycle/emergency-stop/config-version fields
   anywhere in it. `TELEMETRY.md` §8 independently confirms: "Heartbeat-based rapid-suspend...
   nothing in this pipeline currently halts a running agent loop." Treat the "heartbeat rides the
   telemetry response" idea as vision-stage, not implemented.
4. **PRD/TRD's framing of `Matimo-Agent-Signature` as covering "AGDK's check-in calls" (originally
   just tools/telemetry/heartbeat) undersells what actually got built**: D18 (2026-09-11) extended
   mandatory-when-enabled signing to `/v1/chat/completions`/`/v1/messages` too, ahead of the
   originally-scheduled M2b slot. The §1 route table and §5.2 enforcement table reflect the real,
   current posture; older doc language describing signing as tools/telemetry-only is outdated.
5. **Anthropic `stop_reason: "content_filter"`** (§6.3) is flagged in the code's own comment as not
   matching Anthropic's real public API enum , this is a disclosed, not-yet-resolved gap between
   TRD §3b's decision and the live third-party API, not a doc/code mismatch to fix in AGDK; AGDK
   should just be aware the value may look unfamiliar to strict Anthropic-SDK-shaped consumers.
6. **The legacy `/api/v1/enterprise/agents/register`'s `externalFramework` validation is looser**
   (`min(1).max(100)` free string) than `/v1/identities`'s closed four-value enum , a doc reading
   "registration validates externalFramework" without specifying which endpoint could mislead a
   client into assuming the same enum applies everywhere. It does not.
