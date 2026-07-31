# DUM-E Concurrency Rewrite — PR-7 (Serve/Train Split) + PR-8 (Windowed Coalescing)

*Design doc. All anchors re-grepped against the CURRENT tree on 2026-07-31 (post PR-1..PR-6). Where the older TRAINING_CONCURRENCY_PLAN cited a line number, that number has DRIFTED — the anchors below are the live ones. Every "[GPU]" tag marks a claim that cannot be verified without a live A10G.*

Two independent-but-ordered PRs:

- **PR-7** splits `modal_app/trainer.py` into a read-only `Server` pool + a single-writer `Trainer`, adds an immutable version-keyed serve cache, and repoints every reader call site. Ships first.
- **PR-8** replaces the one-job-per-lesson worker with a coalescing batch loop gated by a single-owner writer lease, with token-scaled budget and bisection poison isolation. Depends on PR-7 (needs reads off the writer so a long coalesced train can't freeze chat).

Invariants preserved throughout: ONE SHARED BRAIN (single atomically-advanced CURRENT pointer, single writer); UNRESTRICTED (no content gating added); ROBUSTNESS (validate-before-flip + `LAST_GOOD` + 3-tier reader + non-empty chat all kept); and the corruption fix that forced `max_inputs=1` — writes stay on a single-writer `Trainer`, reads move to an immutable `Server`.

---

## 0. Current-state anchors (verified 2026-07-31)

**`modal_app/trainer.py`** — ONE class `Trainer`:
- `@app.cls(... gpu="A10G", scaledown_window=300, max_containers=1)` at **225-234**; `@modal.concurrent(max_inputs=1)` at **235**.
- `load()` `@modal.enter()` **262**: builds `self.model` (**284**) + `self.base_state` CPU clone (**290**).
- Mutating helpers: `_unwrap_peft` **296**, `_materialize_current` **343** (assigns `self.model`), `_reset_base` **392** (in-place `load_state_dict`).
- Train: `finetune` **467** → `_finetune_inner` **530**; finite-loss guard **676**, merge **687-700**, save **703-710**, smoke-generate **741-768**, `_write_last_good` **773**, `_flip_current` **776**, `vol.commit()` **777**.
- Read (TO REMOVE from Trainer): `generate` **852**, `generate_stream` **898**; both call `_load_current_model` **1113**.
- Read helpers: `_try_load_version` **1059** (full **1076-1092**; legacy adapter merges into `self.model` **1100-1111**), `_load_current_model` **1113** (3-tier: CURRENT→LAST_GOOD→base, `vol.reload()` **1135**, `_reset_base` at tier-3 **1162**), `_build_input_ids` **1165**, `_cleanup_loaded` **1179**.
- Write ops staying on Trainer: `consolidate` **794**, `prune_versions` **947**, `set_current` **986**, `read_current` **1006**, `reset_memory` **1016**.
- Module helpers: `_flip_current` **180**, `_read_current` **189**, `_write_last_good` **198**, `_read_last_good` **213**; `reset_weights` standalone fn **1197**; local entrypoint `reset()` **1278** (calls `Trainer().reset_memory` **1291**).

**`backend/app/training.py`**:
- `_lookup_trainer` **140** (`modal.Cls.from_name(MODAL_APP_NAME, "Trainer")`).
- Readers: `_generate_remote` **153**, `warmup` **175**, `infer` **291**, `infer_chat` **296**, `infer_chat_stream` **306** — all resolve `Trainer` and call `generate`/`generate_stream`.
- Writer path: `_iter_remote_gen` **345**, `_augment_and_guard` **490**, `_build_replay_buffer` **449**, `_run_job` **556** (resolve CURRENT **605-607**, replay+train **621-630**, flip DB **642-646**), `_run_consolidation` **726**, `_worker_loop` **799** (claims ONE job via `db.claim_next_job` **810-811**), `start_worker` **832**, `stop_worker` **854**.
- `enqueue_lesson` **403**, `enqueue_lesson_augment` **419**, `enqueue_consolidation` **689**.
- `WORKER_ID` **49**.

**`backend/app/db.py`**:
- `_connect` **36** with `PRAGMA busy_timeout=5000` **56**.
- `training_jobs` schema **146-158** (has `job_kind`, `claimed_by`, `claimed_at`, `attempts`); index **159**.
- `claim_next_job` **696** (`BEGIN IMMEDIATE` **713**), `finish_job` **748**, `requeue_job` **765**, `recover_stale_jobs` **783**.
- `get_current_weights` **612**, `new_weights_version` **563**, `set_current_weights` **594**, `get_allowed_pairs_since` **397** (dedupe keeps FIRST seen **449-456**).

**`backend/app/main.py`**: chat_stream `event_gen` **345** streams via `training.infer_chat_stream` **379**; `create_lesson` **502** enqueues via `enqueue_lesson_augment` **640**; `warmup` **731-739**; `admin_reset` **856** (drain→`reset_remote`→`clear_learning_state`→`clear_broadcasters`→restart); `/api/health` **959** reads `read_current_version` **987**.

**`backend/app/config.py`**: `MODAL_APP_NAME` **209**; `TRAIN_POLL_INTERVAL` **157**; `TRAIN_JOB_MAX_ATTEMPTS` **158**; `REPLAY_BUFFER_MAX` **169**; `LESSON_KIND_KNOBS` **104**; `KIND_DEFAULTS` **122**; `CONSOLIDATE_KEEP_VERSIONS` **190**; `MAX_NEW_TOKENS` **149**; `MODEL_CONTEXT` **148**. **NOTE:** `MAX_TRAIN_SECONDS` (25.0) and `CONSOLIDATE_MAX_SECONDS` (180.0) live ONLY as trainer.py module constants (**66**, **133**), not in backend config — the backend has no direct knowledge of them today; PR-8 must add mirrors (§B.7).

---

# PART A — PR-7: Serve/Train split

## A.1 Target shape

Two `@app.cls` in the SAME `app = modal.App("unrestricted-ai")`, sharing the same `vol`, `hf_cache`, `image`, secret:

```python
@app.cls(
    image=image, gpu="A10G",
    volumes={"/weights": vol, "/root/.cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name("huggingface-token")],
    min_containers=1,        # keep-warm serve pool (S3): kills ~1s cold-load
    max_containers=4,        # N replicas; scale to load
    scaledown_window=300,
)
@modal.concurrent(max_inputs=6)   # M>1: reads no longer mutate self.model
class Server:
    """READ-ONLY. generate + generate_stream. Immutable base + version-keyed cache."""

@app.cls(
    image=image, gpu="A10G",
    volumes={"/weights": vol, "/root/.cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name("huggingface-token")],
    max_containers=1,        # single writer owns the pointer flip
    scaledown_window=300,
)
@modal.concurrent(max_inputs=1)   # serialize writes on the one warm container
class Trainer:
    """WRITE-ONLY. finetune + consolidate + set_current + prune_versions
       + read_current + reset_memory. generate/generate_stream REMOVED."""
```

Both classes live in `trainer.py` (filename unchanged; the backend looks them up by class name, not import). `min_containers=1`/`max_containers=4`/`max_inputs=6` are starting points; the OOM-critical property is the per-replica reload lock (§A.4), not the exact numbers. **[GPU]** exact safe `max_inputs` and replica count need a VRAM measurement (base ~2.5 GB bf16 + KV; a version reload transiently holds base + old-served + new = ~3 checkpoints).

## A.2 The immutability requirement (the blocker)

Today every reader path can assign `self.model` or call `_reset_base()` (in-place `load_state_dict`): `_load_current_model` tier-3 (**1162**), `_try_load_version` legacy branch (**1100, 1105, 1108**), and `_materialize_current` (**366, 375, 382, 384, 389**). Under `max_inputs>1` those mutations race a concurrent `forward()` on the SAME module object → corruption. So the Server must NEVER touch a shared attribute on the read path.

**Rule:** on `Server`, `self.base_model` (built once in `load()`) is READ-ONLY and only ever used as the tier-3 fallback *and* as the frozen base for legacy adapter loads (via a local `copy.deepcopy`, never in place). Every served model is a LOCAL variable returned to the caller and freed after the call. Nothing on a read path assigns `self.model` (there is no `self.model` on Server at all) or calls anything resembling `_reset_base`.

## A.3 `Server` class sketch

```python
class Server:
    @modal.enter()
    def load(self) -> None:
        import torch, threading
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.dev = "cuda"
        self.tok = AutoTokenizer.from_pretrained(BASE_MODEL)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        # Always-resident, NEVER-mutated pristine base (tier-3 fallback + legacy base).
        self.base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.bfloat16
        ).to(self.dev)
        self.base_model.eval()
        for p in self.base_model.parameters():
            p.requires_grad_(False)
        # Version-keyed serve cache (S1). Exactly one entry: the model for the
        # version string it was loaded from. Guarded by a per-replica lock so N
        # concurrent callers that all detect a new pointer don't from_pretrained
        # simultaneously (N x VRAM -> OOM).
        self._served_model = None          # nn.Module or None (None => serve base)
        self._served_version = None        # str | None  (the version _served_model is)
        self._reload_lock = threading.Lock()
        self._last_reload_ts = 0.0         # throttle vol.reload()

    # ---- immutable resolution: returns a LOCAL model, never mutates shared attrs
    def _current_served(self):
        """Return (model, tok) for CURRENT, reloading the cache only on a pointer move.

        3-tier immutable: CURRENT -> LAST_GOOD -> base. Each tier resolves to a
        module WE RETURN; we never assign a shared attribute except the guarded
        cache swap in _refresh_to(), and that swap installs a fully-built model.
        """
        self._maybe_reload_volume()
        current = _read_current()
        last_good = _read_last_good()

        # Fast path: pointer unchanged and we already hold that version. No lock,
        # no reload -> this is the S1 win (~0.71-0.84s/turn removed).
        if current is not None and current == self._served_version and self._served_model is not None:
            return self._served_model, self.tok

        # Pointer moved (or first serve): serialize the (expensive) reload.
        target = current or last_good  # try CURRENT, then LAST_GOOD
        model = self._refresh_to(target, current, last_good)
        return model, self.tok

    def _refresh_to(self, target, current, last_good):
        # If someone else already refreshed to CURRENT while we waited, use it.
        with self._reload_lock:
            if current is not None and self._served_version == current and self._served_model is not None:
                return self._served_model
            # Build the new served model in a LOCAL first; only swap the cache
            # attribute once it's fully loaded (atomic pointer swap, no partial).
            for version in _dedupe([current, last_good]):
                built = self._build_served(version)   # returns module or None
                if built is not None:
                    old = self._served_model
                    self._served_model = built         # atomic attr rebind
                    self._served_version = version
                    self._free(old)                    # drop the previous cache
                    return built
            # Tier 3: no usable version -> serve the resident base (do NOT cache
            # it as a version; leave _served_version None so a later flip reloads).
            old = self._served_model
            self._served_model, self._served_version = None, None
            self._free(old)
            return self.base_model

    def _build_served(self, version):
        """Load ONE version into a fresh local module (full ckpt or legacy adapter).
        Returns the module or None on any failure. NEVER mutates self.base_model."""
        import torch, copy
        if not version or not os.path.isdir(os.path.join(WEIGHTS_DIR, version)):
            return None
        ver_dir = os.path.join(WEIGHTS_DIR, version)
        kind = self._version_meta(version).get("kind", "full")
        try:
            if kind == "full":
                from transformers import AutoModelForCausalLM
                m = AutoModelForCausalLM.from_pretrained(ver_dir, torch_dtype=torch.bfloat16).to(self.dev)
            else:
                # Legacy adapter: merge onto a DEEPCOPY of base, never base itself.
                from peft import PeftModel
                if not self._is_lora_adapter_dir(version):
                    return None
                base_copy = copy.deepcopy(self.base_model)
                m = PeftModel.from_pretrained(base_copy, ver_dir).merge_and_unload()
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)
            return m
        except Exception:
            logging.warning("Server._build_served(%s) failed", version, exc_info=True)
            return None

    def _maybe_reload_volume(self):
        # Throttle vol.reload(): at most once per RELOAD_THROTTLE_S. A flip commits
        # a new CURRENT; we don't need to see it within the same second on every
        # token. This bounds volume metadata churn under max_inputs=6.
        now = time.time()
        if now - self._last_reload_ts < RELOAD_THROTTLE_S:
            return
        try:
            vol.reload(); self._last_reload_ts = now
        except Exception:
            logging.warning("Server vol.reload() failed", exc_info=True)

    def _free(self, m):
        import torch
        if m is not None and m is not self.base_model:
            del m
            torch.cuda.empty_cache()

    @modal.method()
    def generate(self, prompt=None, max_new_tokens=512, messages=None,
                 do_sample=GEN_DO_SAMPLE, temperature=GEN_TEMPERATURE,
                 top_p=GEN_TOP_P, repetition_penalty=GEN_REPETITION_PENALTY) -> str:
        import torch
        model, tok = self._current_served()      # NO per-call cleanup: cache-owned
        ids = self._build_input_ids(tok, prompt, messages)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_new_tokens,
                                  **self._gen_kwargs(tok, do_sample, temperature, top_p, repetition_penalty))
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

    @modal.method()
    def generate_stream(self, prompt=None, max_new_tokens=512, messages=None, ...):
        # Identical body to today's Trainer.generate_stream (898-945) EXCEPT:
        #   model, tok = self._current_served()   # was self._load_current_model()
        #   NO finally: self._cleanup_loaded(...)  -> the cache owns the model
        ...
```

**Assertion to add (guards the invariant in code):** at the top of `generate`/`generate_stream`, after resolution, `assert not hasattr(self, "model")` is not enough — instead a static rule enforced in review: *no Server method assigns `self.base_model`, and no Server method calls `_reset_base`/`_materialize_current` (those don't exist on Server).* Add a unit assertion in `_build_served` that the returned module `is not self.base_model` unless it's the explicit tier-3 return.

### Concurrency safety of the cache under `max_inputs=6`
- Reads of `self._served_model`/`self._served_version` on the fast path are a plain attribute read of a fully-built module — safe to share across the 6 concurrent forwards (inference is read-only w.r.t. weights).
- The only writer is `_refresh_to` under `self._reload_lock`. It builds into a local and does ONE atomic attribute rebind, so a concurrent fast-path reader sees either the whole old model or the whole new one — never a half-built module. `torch` is fine with several `forward()`s on one frozen module concurrently (Python threads; the GIL + CUDA streams serialize kernel launches). **[GPU]** confirm no throughput cliff at `max_inputs=6` on one A10G.
- The freed old model (`self._free(old)`) can still be in use by an in-flight forward that captured the local reference before the swap — that's why we swap the ATTRIBUTE but each caller holds its own local `model` from `_current_served()`; `del m` only drops OUR name, and Python GC won't collect while another frame holds it. `empty_cache()` is advisory. Safe.

## A.4 Per-replica reload lock (the second blocker) — already in the sketch

`self._reload_lock` (a `threading.Lock`, because Modal runs concurrent inputs as threads within a replica) serializes `_refresh_to`. First caller to see a moved pointer takes the lock and loads; the other 5 either (a) find `_served_version == current` already updated and return the new cache, or (b) if they entered `_refresh_to` before the first finished, they block on the lock, then re-check and return the freshly built model. Crucially, WHILE the first is loading, fast-path callers (§A.3 fast path) keep returning the OLD `self._served_model` — the old version is still valid to serve, so no request stalls on the reload. This is what prevents 6× `from_pretrained` → 6× transient VRAM → OOM.

`RELOAD_THROTTLE_S` (new module constant, ~2.0s) throttles `vol.reload()` so 6 concurrent streams don't each hammer volume metadata; a flip becomes visible within ≤ throttle seconds, which is fine (a lesson just landed; sub-2s propagation to replicas is imperceptible).

## A.5 Reader repoint in `backend/app/training.py`

Add a sibling lookup and point every reader at `Server`:

```python
def _lookup_server() -> Any:                       # NEW, next to _lookup_trainer (140)
    return modal.Cls.from_name(settings.MODAL_APP_NAME, "Server")
```

- `_generate_remote` **153**: `trainer_cls = _lookup_trainer()` → `_lookup_server()`; `instance.generate` unchanged (Server has it).
- `infer` **291** / `infer_chat` **296**: unchanged (they call `_generate_remote`).
- `infer_chat_stream` **306**: `_lookup_trainer()` **316** → `_lookup_server()`; `instance.generate_stream` unchanged (Server has it).
- `warmup` **175**: `_lookup_trainer()` **185** → `_lookup_server()`. This now warms the SERVER pool. To warm ≥`min_containers` replicas (not just one), fire `min_containers` concurrent tiny generates with `.spread`/gather, or loop `N` `aio("hi", 1, None)` calls concurrently so Modal fans them across replicas. Add `SERVER_MIN_CONTAINERS` mirror to config so `warmup` knows how many to fan.
- Writer lookups (`set_current_version` **199**, `read_current_version` **221**, `reset_remote` **242**, `prune_versions_remote` **275**, `_iter_remote_gen` **345**, `_iter_consolidate_gen` **699**) all stay on `_lookup_trainer()` — those are writes.

## A.6 Reset must invalidate the Server cache

After a reset the volume CURRENT is gone and `_read_current()` returns None → `_current_served()` tier-3 serves `self.base_model`. But a replica that already holds `_served_model` for the wiped version will keep serving it until it notices the pointer changed. Two coordinated fixes in `reset_remote` (**242**) / `admin_reset` (**856**):

1. **Pointer-driven invalidation (primary):** `reset_weights` (**1197**) removes `CURRENT` and every `v{N}`. On the next serve, `_read_current()` → None and `_read_last_good()` still names a wiped dir → `_build_served` returns None → tier-3 base. So the cache self-invalidates within `RELOAD_THROTTLE_S` of the next request. **BUT** `reset_weights` does NOT remove `LAST_GOOD` today — add `LAST_GOOD` to the removal set in `reset_weights` (**1209-1217**) so tier-2 can't resurrect a wiped version.
2. **Active flush (belt-and-suspenders):** add `Server.flush_cache()` `@modal.method()` that takes `self._reload_lock`, frees `_served_model`, sets `_served_version=None`. `reset_remote` calls it on the Server pool after the volume wipe. Because `max_containers>1`, one call only hits one replica; rely primarily on (1) for the rest, or fan `flush_cache` across `min_containers` like warmup. Document that (1) is authoritative and (2) is a latency optimization.

`reset_memory` (**1016**) stays on `Trainer` (the writer's in-memory model). `reset()` local entrypoint (**1278/1291**) unchanged.

## A.7 Trainer: what to DELETE

- Delete `generate` (**852-896**) and `generate_stream` (**898-945**) from `Trainer`. They move to `Server`.
- Delete `_load_current_model` (**1113-1163**), `_try_load_version` (**1059-1111**), `_cleanup_loaded` (**1179-1190**) from `Trainer` — reader-only helpers. `_build_input_ids` (**1165**), `_gen_kwargs` (**838**), `_version_meta` (**1030**), `_is_lora_adapter_dir` (**1041**) are needed by `Server`, so MOVE them (or hoist to module-level free functions taking `tok`/`dev`, which is cleaner since both classes need them). Recommendation: hoist `_version_meta`, `_is_lora_adapter_dir` to module-level (pure, only touch `WEIGHTS_DIR`); duplicate `_gen_kwargs`/`_build_input_ids` as tiny methods on each class (they need `self.dev`/`self.tok`).
- Trainer keeps `_materialize_current` (**343**) and `_reset_base` (**392**) — those are WRITE-path (build-on-CURRENT before training) and legitimately mutate `self.model`, which is safe under `max_inputs=1`.

## A.8 PR-7 file-by-file

**`modal_app/trainer.py`**
- ADD module constant `RELOAD_THROTTLE_S: float = 2.0` (near line 66).
- ADD `SERVER_MIN_CONTAINERS`/`SERVER_MAX_CONTAINERS`/`SERVER_MAX_INPUTS` constants (or read from decorator literals).
- ADD class `Server` (§A.3): `load`, `_current_served`, `_refresh_to`, `_build_served`, `_maybe_reload_volume`, `_free`, `_gen_kwargs`, `_build_input_ids`, `generate`, `generate_stream`, `flush_cache`.
- HOIST `_version_meta` (**1030**), `_is_lora_adapter_dir` (**1041**) to module-level free functions; update Trainer refs.
- CHANGE `Trainer` decorators: keep `max_containers=1` + `max_inputs=1` (**225/235**).
- DELETE from `Trainer`: `generate` (**852**), `generate_stream` (**898**), `_load_current_model` (**1113**), `_try_load_version` (**1059**), `_cleanup_loaded` (**1179**).
- CHANGE `reset_weights` (**1197**): add `LAST_GOOD` to the wipe set (**1209-1217**), plus the `CURRENT`/`v{N}` it already removes.
- No change to `finetune`/`_finetune_inner`/`consolidate`/`_flip_current`/`_write_last_good` — all PR-1 guards preserved verbatim.

**`backend/app/training.py`**
- ADD `_lookup_server()` (next to **140**).
- CHANGE `_generate_remote` (**160**), `infer_chat_stream` (**316**), `warmup` (**185**) to `_lookup_server()`; `warmup` fans `SERVER_MIN_CONTAINERS` concurrent 1-token generates.
- CHANGE `reset_remote` (**242**) to also call `Server.flush_cache` (fanned) after the volume wipe.
- No change to writer lookups.

**`backend/app/config.py`**
- ADD `SERVER_MIN_CONTAINERS: int = 1`, `SERVER_MAX_CONTAINERS: int = 4`, `SERVER_MAX_INPUTS: int = 6`, `RELOAD_THROTTLE_S: float = 2.0` (mirrors for warmup fan-count + docs; trainer.py keeps its own copies since the container has no backend import).

**`backend/app/main.py`** — no change required (chat already goes through `training.infer_chat_stream`; reset already calls `training.reset_remote`). Optional: `/api/health` (**987**) could also ping Server; leave as-is.

## A.9 PR-7 robustness preservation
- 3-tier CURRENT→LAST_GOOD→base is REBUILT in `Server._refresh_to`/`_build_served` as immutable local resolution (not mutation). Same ordering, same defensiveness (each tier `try/except`→None→next).
- `LAST_GOOD` still written by the writer at flip (**773**); Server reads it. Reset now also wipes it (§A.6) so it can't name a deleted dir.
- Validate-before-flip (finite-loss **676**, smoke-generate **741-768**) is untouched — it's on the writer.
- Non-empty chat fallback lives in `main.py`/`infer_chat_stream` (empty stream → caller falls back); unchanged.

---

# PART B — PR-8: Windowed coalescing

## B.1 Goal & wait-time math

Today `_worker_loop` (**799**) claims ONE `lesson` job per iteration (`claim_next_job` **696**) → one finetune → one flip. Service time per lesson T ≈ `MAX_TRAIN_SECONDS`(25s) + save/flip/replay ≈ ~28s. 50 concurrent teachers FIFO ≈ 50 × 28s ≈ **1400s (~23 min)**.

**Coalescing:** open a window, capture `T0`, claim ALL `lesson` jobs queued before `T0`, union their pairs (newest-wins dedupe), train ONCE, flip ONCE. Each lesson still gets its own row status + feed entry on success.

Wait-time for 50 concurrent teachers, `COALESCE_MAX_PAIRS` cap = 64 union pairs/window, token-scaled budget ≈ 45s/window (see §B.7):
- If 50 lessons ≈ 50 seed-concepts, and each augments to ~90 pairs but we cap the UNION at 64 deduped pairs and process the rest in the next window: 50 lessons / (windows big enough to hold them) → with per-prompt dedupe most windows carry many distinct concepts. Realistic: **2–4 windows** to drain 50 (cap-bound), each ~45–60s → **~1.5–4 min total**, vs 23 min. Chat wait throughout ≈ 0 (Server pool, PR-7).
- Contrast the naive "1 window covers all 50 in 28s" claim in the old plan — REJECTED: a 400-pair union over 6 epochs vastly exceeds 25s. The honest number is 2–4 windows with a token-scaled budget.

| teachers | today (1 job/lesson) | PR-8 coalesced (cap 64, ~45-60s/window) |
|---:|---:|---:|
| 5  | ~140s | 1 window ~45-60s |
| 20 | ~560s | 1-2 windows ~60-120s |
| 50 | **~1400s (~23min)** | **2-4 windows ~90-240s** |

## B.2 The single serialized critical section (unchanged invariant)

`resolve base_version snapshot → union-train → allocate v{N+1} → flip CURRENT + commit`. Everything else (augmentation fanout, replay assembly) parallelizes/precedes it. On any train failure the writer does NOT flip; CURRENT stays last-good; every Server replica keeps answering. This is the ONE-SHARED-BRAIN invariant.

## B.3 Single-owner writer lease (replaces per-claim BEGIN IMMEDIATE as the loop gate)

`BEGIN IMMEDIATE` per-claim guarantees two workers don't claim the SAME row — but with batches, two `_worker_loop`s could claim DISJOINT batches and both resolve `base_version`=vK, then both flip → worker B's flip drops worker A's lessons (B snapshotted the pre-A base). So the whole batch (base-resolution … flip) must be one critical section held by exactly ONE writer.

**Lease row** in a new `writer_lease` table (single row, id=1):

```sql
CREATE TABLE IF NOT EXISTS writer_lease (
    id         INTEGER PRIMARY KEY CHECK (id = 1),
    owner      TEXT,               -- WORKER_ID holding the lease, or NULL
    acquired_at TEXT,
    expires_at TEXT                -- lease TTL; a reaper can steal past this
);
INSERT OR IGNORE INTO writer_lease (id, owner, acquired_at, expires_at)
    VALUES (1, NULL, NULL, NULL);
```

`db.acquire_writer_lease(worker_id, ttl_s) -> bool` (BEGIN IMMEDIATE):
```sql
UPDATE writer_lease
SET owner = ?, acquired_at = ?, expires_at = ?
WHERE id = 1 AND (owner IS NULL OR expires_at < ?);   -- free or expired
-- rowcount==1 => acquired
```
`db.renew_writer_lease(worker_id, ttl_s)` (heartbeat, extends `expires_at` WHERE owner=?), `db.release_writer_lease(worker_id)` (owner=NULL WHERE owner=?).

`_worker_loop` acquires the lease ONCE per batch iteration, renews it via heartbeat during a long train, releases in `finally`. TTL is **from the job's own budget** (§B.7), never a flat 50s. Only the lease-holder resolves base_version + flips → the write-behind-write race is gone even with `--workers>1`.

## B.4 `db.claim_next_batch` (generalizes `claim_next_job`)

New `db.py` function (alongside **696**). Called only by the lease-holder:

```python
def claim_next_batch(worker_id, max_attempts=3, max_pairs=None, window_cutoff_iso=None):
    """Atomically claim a BATCH of queued 'lesson' jobs (BEGIN IMMEDIATE).

    Hard window: only jobs with created_at <= window_cutoff_iso join the batch
    (T0 captured by the caller at window open) so sustained load can't sweep in
    new arrivals forever. Consolidation jobs are NEVER batched (claimed singly by
    the existing path). Bounds the union by summing an estimated pair count until
    it would exceed max_pairs (a coarse cap; augmentation happens later, so we
    cap by job count as a proxy: COALESCE_MAX_JOBS).

    Returns {"batch_id": <uuid>, "jobs": [row,...]} or None. Every claimed row ->
    status='claimed', claimed_by=worker_id, claimed_at=now, attempts+1, and a new
    'batch_id' column stamped so a reaper/finish can address the batch.
    """
    # BEGIN IMMEDIATE
    #   rows = SELECT * FROM training_jobs
    #          WHERE status='queued' AND job_kind='lesson'
    #            AND attempts < ? AND created_at <= ?         -- hard cutoff
    #          ORDER BY id ASC LIMIT ?                        -- COALESCE_MAX_JOBS
    #   if not rows: COMMIT; return None
    #   batch_id = uuid4().hex
    #   UPDATE ... SET status='claimed', batch_id=?, ... WHERE id IN (...)
    # COMMIT
```

Add a `batch_id TEXT` column to `training_jobs` (+ migration `_migrate_training_jobs_batch_id`, mirroring the existing `_migrate_*` pattern at **179-261**). `claim_next_job` (**696**) stays for consolidation jobs (job_kind='consolidate', claimed singly).

`finish_job`/`requeue_job` gain a batch-addressed sibling or a `batch_id` param so all rows in a batch move together on success; on partial failure see §B.6 (bisection). Each lesson row still gets its own `set_lesson_status(... 'done')` + `add_feed` on success (per-lesson feed entries preserved).

## B.5 New worker loop (`training.py`)

Replace `_worker_loop` (**799-829**) with a coalescing loop; `_run_job` (**556**) is refactored into `_run_batch`. Consolidation stays on the single-claim path and is NEVER coalesced.

```python
async def _worker_loop(stop):
    while not stop.is_set():
        # Consolidation first: claimed singly (job_kind='consolidate'), highest
        # priority so a nightly job isn't starved behind lesson windows.
        cjob = await asyncio.to_thread(db.claim_next_consolidation, WORKER_ID, MAX_ATTEMPTS)
        if cjob is not None:
            await _run_consolidation(cjob)          # existing path (726), unchanged
            continue

        # Lesson batch: acquire the single-owner writer lease for the batch.
        ttl = _lease_ttl_for("lesson")              # budget-derived, §B.7
        if not await asyncio.to_thread(db.acquire_writer_lease, WORKER_ID, ttl):
            await _sleep_or_stop(stop); continue     # another worker owns it
        try:
            t0 = db._now()                           # hard window cutoff T0
            batch = await asyncio.to_thread(
                db.claim_next_batch, WORKER_ID, MAX_ATTEMPTS,
                settings.COALESCE_MAX_PAIRS, t0)
            if batch is None:
                await _sleep_or_stop(stop); continue
            await _run_batch(batch, stop)            # heartbeats the lease inside
        finally:
            await asyncio.to_thread(db.release_writer_lease, WORKER_ID)
```

`_run_batch(batch, stop)`:
1. For each job: if envelope (`_AUGMENT_ENVELOPE_KEY`), run `_augment_and_guard` (**490**) → per-lesson allowed pairs (blocked lessons finish here, per-lesson status). Bare-list legacy jobs pass through. (Augmentation is per-lesson and can run concurrently with a bounded gather — the OpenRouter semaphore already bounds it.)
2. **Union + newest-wins-per-prompt dedupe (USER DECISION):** build a dict keyed by normalized `prompt`; iterate jobs in ascending lesson-id and overwrite, so the HIGHEST lesson-id's response wins per prompt. This is the same-window contradiction rule — never a blended {3,5}. Contrastive/anchor pairs keep distinct prompts so they aren't collapsed.
3. Resolve `base_version` ONCE under the lease: `current = db.get_current_weights()` (**605**), `base_version = current["path"]`, `parent_id = current["id"]`.
4. Build replay buffer over the UNION (`_build_replay_buffer` **449**, dedup against the union's pairs) → `payload = replay + union`.
5. Stream ONE `Trainer.finetune` (`_iter_remote_gen` **345**) with a token-scaled `max_train_seconds` knob (§B.7) — requires plumbing a `max_train_seconds` override through `finetune`/`_finetune_inner` (it already accepts `max_train_seconds` at **541**, but `finetune` **467** doesn't expose it; ADD the kwarg to `finetune` and to `_iter_remote_gen`'s `knobs`). Heartbeat `renew_writer_lease` on each progress event (or every K events) so a long union doesn't let the reaper steal the lease.
6. On terminal `done`: `new_weights_version` + `set_current_weights` ONCE (the single flip). Then for EACH lesson in the batch: `set_lesson_status(lid,'done')`, `add_feed(lid, ...)`, and mark its job done (batch-addressed `finish_job`). The trainer already flipped the volume CURRENT + committed (**776-777**); the DB flip mirrors it once.
7. On failure: bisection (§B.6), not fan-out-to-singles.

Which lesson "owns" the produced `v{N}`? The version's `lesson_id` is ambiguous for a union. Decision: set `weights_versions.lesson_id = NULL` for a coalesced batch (like consolidation), and record batch membership via the per-lesson feed entries + a new optional `batch_id` on the version row (or leave lesson_id as the highest id for `get_weights_version_by_lesson` **475** to keep working for train-status reads). Recommendation: set `lesson_id` to the batch's max lesson-id so `/api/train/status/{lesson_id}` (**682**) resolves for at least that lesson; document that coalesced siblings read status from their own `lessons.status='done'` row, not the version row.

## B.6 Poison isolation by bisection (not fan-out)

A batch that OOMs/diverges must not fan out to N sequential singles (that reverts to the 23-min FIFO). Instead:
- On a batch train failure that is CLASSIFIED TRANSIENT (not OOM/CUDA), split the batch's jobs into two halves, requeue each half as-is (they'll re-batch), and let the loop retry — ~log₂(N) passes isolate the bad job.
- `_classify_failure(exc)`: OOM/`CUDA error`/`device-side assert` → **TERMINAL** (do NOT requeue-storm): mark every job in the failing SINGLETON terminal `error`; but for a MULTI-job batch, an OOM is likely one oversized lesson → bisect to find it, and only the singleton that still OOMs is marked terminal. Bound pair text length (~2KB) at enqueue (`create_lesson`/`enqueue_lesson_augment`) so one pasted wall-of-text can't OOM.
- A job's `attempts` still bounds retries (`TRAIN_JOB_MAX_ATTEMPTS`); a job that has bisected down to a singleton and failed `max_attempts` is terminal. No requeue-storm: bisection halves, it doesn't multiply.

## B.7 Token-scaled budget + lease TTL

The union can be large; 25s does NOT cover 400 pairs. Two levers (use BOTH):
- **Cap the union:** `COALESCE_MAX_PAIRS` (≈64) and `COALESCE_MAX_JOBS` (≈16) bound one window; overflow drains in the next window. Accept 2–4 windows.
- **Scale the time budget:** pass `max_train_seconds = min(COALESCE_MAX_TRAIN_SECONDS, base + per_pair * n_union_pairs)` to `finetune`. e.g. `base=15`, `per_pair=0.4`, cap `90` → a 64-pair union gets ~41s, a 16-pair window ~21s. **[GPU]** the `per_pair` constant needs one measured A10G run to calibrate; the plan's "28s covers 400 pairs" is explicitly wrong.

**Lease TTL** = job-budget-derived, NEVER a flat 50s:
- lesson batch: `ttl = COALESCE_MAX_TRAIN_SECONDS + AUGMENT_SLACK(~60s for fanout) + COLD_START_SLACK(~90s) + FLIP_SLACK(~15s)` ≈ up to ~255s, renewed by heartbeat so a live batch never expires.
- consolidation is claimed on its own single path with its own lease acquisition: `ttl = CONSOLIDATE_MAX_SECONDS(180) + COLD_START_SLACK(90) + FLIP_SLACK(15)` ≈ 285s.
- The reaper (§B.8) only steals a lease whose `expires_at < now`; heartbeats keep a live batch's lease fresh, so the reaper can't kill a running consolidation or a big window.

New config knobs (`config.py`, near **156-169**): `COALESCE_MAX_PAIRS=64`, `COALESCE_MAX_JOBS=16`, `COALESCE_BASE_SECONDS=15.0`, `COALESCE_PER_PAIR_SECONDS=0.4`, `COALESCE_MAX_TRAIN_SECONDS=90.0`, `WRITER_LEASE_AUGMENT_SLACK_S=60`, `WRITER_LEASE_COLDSTART_SLACK_S=90`, `WRITER_LEASE_FLIP_SLACK_S=15`, `MAX_PAIR_TEXT_BYTES=2048`. Also ADD backend mirrors `MAX_TRAIN_SECONDS=25.0`, `CONSOLIDATE_MAX_SECONDS=180.0` (today only trainer.py knows these; the lease-TTL math needs them backend-side).

## B.8 Reaper (stale batch / dead worker)

`db.reap_stale_leases_and_jobs()` (extends `recover_stale_jobs` **783**), called on startup AND periodically by the loop:
- If `writer_lease.expires_at < now`: the holder is dead/wedged → NULL the owner (steal), and requeue every `training_jobs` row still `claimed` by that dead owner (status→queued, batch_id→NULL) so a live worker re-batches them. Attempts still bound retries.
- Add a `claimed_at`-age fallback for jobs whose worker died WITHOUT the lease (belt-and-suspenders): `claimed` age > (lease max TTL) → requeue.
- Consolidation `claimed` rows older than its TTL likewise requeue (unless it's a live heartbeated lease).

## B.9 Consolidation stays isolated (unchanged invariant)
- `_run_consolidation` (**726**) is NEVER coalesced. It's claimed on its own single path (`db.claim_next_consolidation`, a thin filter of `claim_next_job` for `job_kind='consolidate'`), acquires the writer lease with a consolidation-sized TTL, resolves base=None, trains the full corpus, flips once. Priority over lesson batches so the nightly job isn't starved.
- `get_allowed_pairs_since` (**397**) currently dedups keeping the FIRST-seen response (**449-456**) — this loses a same-day override (the m6 bug). Since PR-8 makes newest-wins the batch rule, ALSO fix consolidation dedupe to keep the LATEST response per prompt (order by lesson id / feed time DESC, keep first). Small change in `get_allowed_pairs_since` — worth doing here so consolidation doesn't silently undo the newest teach that the live path honored.

## B.10 PR-8 file-by-file

**`backend/app/db.py`**
- ADD `writer_lease` table to `_SCHEMA` (near **146**) + `INSERT OR IGNORE` seed row.
- ADD `batch_id TEXT` column to `training_jobs` + migration `_migrate_training_jobs_batch_id` (pattern of **216-261**); call it in `init_db` (**164-176**).
- ADD `acquire_writer_lease`, `renew_writer_lease`, `release_writer_lease`, `reap_stale_leases_and_jobs`.
- ADD `claim_next_batch` (§B.4) and `claim_next_consolidation` (single-claim filter for `job_kind='consolidate'`).
- ADD batch-addressed `finish_batch_jobs(batch_id, status)` / extend `requeue_job` to accept a list or batch_id.
- CHANGE `get_allowed_pairs_since` (**397**) dedupe to keep the LATEST response per prompt (m6 fix, §B.9).
- KEEP `claim_next_job` (**696**) for compat/consolidation.

**`backend/app/training.py`**
- REPLACE `_worker_loop` (**799**) with the coalescing loop (§B.5); ADD `_run_batch`, `_lease_ttl_for`, `_classify_failure`, `_sleep_or_stop`, `_bisect_and_requeue`.
- REFACTOR `_run_job` (**556**) body into `_run_batch` (per-lesson augment/guard loop → union dedupe → single resolve/train/flip → per-lesson status+feed).
- CHANGE `_iter_remote_gen` (**345**) to plumb a `max_train_seconds` override into `knobs`.
- ADD lease heartbeat calls inside the `async for event` loop.
- `_run_consolidation` (**726**) unchanged except it now acquires/heartbeats/releases the writer lease around its critical section, and is dispatched from the new loop's consolidation-first branch.
- `start_worker` (**832**) calls `db.reap_stale_leases_and_jobs()` (superset of `recover_stale_jobs`).

**`modal_app/trainer.py`**
- CHANGE `finetune` (**467**) signature to EXPOSE `max_train_seconds` (it's already accepted by `_finetune_inner` at **541**; just pass it through at **516-519**). No other trainer change — PR-1 guards (finite-loss **676**, smoke **741**, LAST_GOOD **773**) untouched.

**`backend/app/config.py`**
- ADD the coalescing/lease/budget knobs (§B.7) + backend mirrors of `MAX_TRAIN_SECONDS`/`CONSOLIDATE_MAX_SECONDS`.

**`backend/app/main.py`**
- CHANGE `create_lesson` (**502**) / `enqueue_lesson_augment` path to enforce `MAX_PAIR_TEXT_BYTES` on seed pairs before enqueue (poison bound, §B.6). No other change — the queue contract is unchanged from the endpoint's view.

## B.11 PR-8 robustness preservation
- Validate-before-flip, LAST_GOOD, 3-tier reader: entirely on the writer/Server, untouched. A coalesced union that diverges hits the finite-loss guard (**676**) → no flip → CURRENT stays last-good → Server keeps serving.
- Single flip per batch preserves ONE-SHARED-BRAIN: exactly one `new_weights_version`+`set_current_weights` per window, under the single-owner lease.
- Newest-wins-per-prompt dedupe (USER DECISION) is deterministic; no {3,5} blend.
- Bisection bounds poison; OOM/CUDA terminal classification stops requeue-storms; `attempts` still caps retries.
- Per-lesson `lessons.status` + `learned_feed` rows preserved on success (each teacher sees their lesson land).

---

## C. Ordering, testing, and what needs a live GPU

**Order:** PR-7 first (reads off the writer), then PR-8 (a coalesced batch can now train for 45–90s without freezing chat). PR-8 without PR-7 would reintroduce read-behind-write for the whole window.

**Isolated ephemeral testing (USER DECISION — no push, no redeploy over live app):**
- All Modal tests use throwaway scripts defining an EPHEMERAL app (`modal.App("dume-pr7-test-<uuid>")`) with copies of `Server`/`Trainer`, run via `modal run scratchpad/...`. Never touch the deployed `unrestricted-ai` app.
- PR-8 DB/lease logic (`claim_next_batch`, lease acquire/renew/steal, bisection, newest-wins dedupe) is pure SQLite + Python → unit-testable LOCALLY with a temp DB, no GPU. Test: two simulated workers race for the lease; disjoint-batch flip race cannot occur; hard cutoff excludes post-T0 arrivals; newest-wins keeps highest lesson-id per prompt.

**Cannot be verified without a live A10G [GPU]:**
1. Safe `Server` `max_inputs` / replica count before a throughput cliff or VRAM OOM (base + old-served + new-served transient during a reload).
2. That `max_inputs=6` frozen-module concurrent `forward()`s don't corrupt/contend on one A10G (the whole premise of the split).
3. The S1 cache win magnitude (plan says ~0.71–0.84s/turn) on the current 1B full-checkpoint reload.
4. `COALESCE_PER_PAIR_SECONDS` calibration for the token-scaled budget (the "28s per 400 pairs" figure is wrong; needs one measured run).
5. Actual per-window wall-clock for 50 teachers (the 2–4 window / 90–240s estimate).
6. That `_refresh_to`'s "serve OLD cache while first caller reloads" holds under real concurrent flip + serve (no 6× `from_pretrained` OOM).

**Definitely-true-without-GPU (pure logic):** the lease eliminates the disjoint-batch flip race; the hard T0 cutoff prevents infinite window sweep; newest-wins is deterministic; bisection is log₂; the immutable Server never assigns a shared weight attribute on the read path.
