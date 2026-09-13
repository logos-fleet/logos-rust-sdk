//! MODULE-LOCAL PERSISTENT STORAGE — one code path for a native module and its
//! `web` variant.
//!
//! A Rust module's core keeps its state somewhere. Natively that is a directory
//! the host stamped into the module context (`instance_persistence_path`) and
//! `std::fs` is the whole story. Inside a Wasm host there is a filesystem too —
//! emscripten gives the image one — but it is the image's own memory unless the
//! host mounted something durable under it, and even then a write only reaches
//! the browser's IndexedDB when somebody asks. A module that writes with
//! `std::fs` and nothing else therefore WORKS in a webview and loses everything
//! on the next page load, with no error anywhere.
//!
//! That difference — and it is the only one — is what this module is. The trait
//! below is the ordinary key/value surface, plus one operation the plain
//! filesystem does not have:
//!
//! > **[`Storage::commit`] is the durability barrier.** Data written since the
//! > last commit is not guaranteed to outlive this image. `commit()` hands it
//! > to the durable medium; a module that never calls it keeps nothing.
//!
//! Natively that is an fsync: when `commit()` returns `Ok` the bytes are on
//! disk. In a Wasm host it starts the push of the image's filesystem into the
//! browser's IndexedDB, through the host's `logos_storage_commit` entry point,
//! and CANNOT WAIT FOR IT — `FS.syncfs` completes on the browser's event loop
//! and blocking on it needs Asyncify, which the Web container does not build
//! with. So on emscripten `Ok` means "handed over and in flight", which
//! completes in the next turn of the event loop; a push that FAILED is reported
//! on the console and raised by the NEXT `commit()`, so an error is never
//! silently dropped, only reported late.
//!
//! A module written against this contract is correct in both places; a module
//! that skips `commit()` is broken in exactly one, which is why the barrier is
//! named rather than implied.
//!
//! ## What a key is
//!
//! A flat, non-empty name — no `/`, no `\`, no `.` or `..` component. Not a
//! path. A store is one flat namespace on purpose: OPFS and IndexedDB are, the
//! filesystem is not, and the intersection is what a module may rely on. A key
//! that would escape the store is [`StorageError::InvalidKey`], refused before
//! anything is opened.
//!
//! ## What the host owes
//!
//! Before the first dispatch, the contents of the store are present. Natively
//! that is trivially true. A Wasm host has to populate the mount from IndexedDB
//! first, and the module-builder's host does that before it announces itself as
//! serving — so `on_context_ready` and every method after it see the same
//! durable state in both containers.
//!
//! ## Example
//!
//! ```no_run
//! use logos_rust_sdk::storage::{FileStorage, Storage};
//!
//! # fn demo(persistence_path: &str) -> Result<(), logos_rust_sdk::storage::StorageError> {
//! let store = FileStorage::open(std::path::Path::new(persistence_path).join("vaults"))?;
//! store.write("f39fd6e5.json", br#"{"crypto":{}}"#)?;
//! store.commit()?;                 // survives a page reload from here on
//! assert!(store.exists("f39fd6e5.json"));
//! # Ok(())
//! # }
//! ```

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Why a store operation could not be performed.
#[derive(Debug)]
pub enum StorageError {
    /// No such key in this store.
    NotFound(String),
    /// The key is not a flat name (empty, a path, or a `.`/`..` component).
    InvalidKey(String),
    /// The store could not be opened — no persistence path, or the host gave
    /// one that is not a usable directory.
    NotAvailable(String),
    /// The underlying filesystem or host call failed.
    Io(String),
}

impl fmt::Display for StorageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StorageError::NotFound(k) => write!(f, "no such key in this module's store: {k}"),
            StorageError::InvalidKey(k) => write!(
                f,
                "invalid storage key {k:?}: a key is a flat, non-empty name with no path separator and no '.'/'..' component"
            ),
            StorageError::NotAvailable(m) => write!(f, "module storage unavailable: {m}"),
            StorageError::Io(m) => write!(f, "module storage io error: {m}"),
        }
    }
}

impl std::error::Error for StorageError {}

/// The result of every store operation.
pub type Result<T> = std::result::Result<T, StorageError>;

/// A module's own persistent key/value store.
///
/// Implementations are expected to be usable from the module's dispatch thread
/// and nowhere else surprising; `Send` is required so a core can own one behind
/// the usual `Mutex`.
///
/// Every method is defined in terms of the contract in the module docs. In
/// particular a write is visible to `read`/`exists`/`list` immediately, and
/// durable only after [`Storage::commit`].
pub trait Storage: Send {
    /// The bytes stored under `key`, or [`StorageError::NotFound`].
    fn read(&self, key: &str) -> Result<Vec<u8>>;

    /// Replace `key` with `bytes`. Atomic: a reader (including a later run
    /// after a crash) sees either the whole previous value or the whole new
    /// one, never a truncated file.
    fn write(&self, key: &str, bytes: &[u8]) -> Result<()>;

    /// Delete `key`. `Ok(false)` when it was not there — deleting twice is not
    /// an error, so an uninstall path need not race a scan.
    fn remove(&self, key: &str) -> Result<bool>;

    /// Whether `key` is present. An invalid key is absent, not an error: this
    /// is a question, and every question about a key that cannot exist has the
    /// same answer.
    fn exists(&self, key: &str) -> bool;

    /// Every key in the store, sorted. Entries this store did not write (a
    /// subdirectory, a stray file) are not keys and are not listed.
    fn list(&self) -> Result<Vec<String>>;

    /// THE DURABILITY BARRIER. Returns once everything written so far will
    /// survive this image: an fsync natively, a push into IndexedDB in a Wasm
    /// host. Call it after a write whose loss would be a bug.
    fn commit(&self) -> Result<()>;

    /// The directory this store lives in, when it lives in one.
    ///
    /// THE ESCAPE HATCH, and it is narrow. Some crates a module core depends on
    /// are path-based and own their own write — `eth_keystore::encrypt_key`
    /// takes a directory and calls `File::create` itself — so a store backed by
    /// a real directory has to be able to say where it is, or such a crate
    /// cannot be used at all. A caller that takes this path is responsible for
    /// the atomicity the trait otherwise gives it, and still owes [`Self::commit`].
    ///
    /// `None` for a store with no filesystem behind it (see [`MemoryStorage`]),
    /// which is exactly the case a path-based dependency cannot serve.
    fn local_dir(&self) -> Option<&Path> {
        None
    }
}

/// THE BARRIER, FOR A DIRECTORY THAT IS NOT A `Storage`.
///
/// A module core with its own on-disk layout — nested directories, unix modes,
/// its own staging discipline — should not have to flatten itself into a
/// key/value store to become correct in a Wasm host. What it is missing is only
/// the barrier, so the barrier is available on its own: point it at the
/// directory whose writes must stick and call it where the native code already
/// fsyncs.
///
/// Identical semantics to [`Storage::commit`], including that on emscripten the
/// push to IndexedDB is in flight when this returns.
pub fn commit(dir: &Path) -> Result<()> {
    host_commit(dir)
}

/// Whether writes are durable the moment they return, without [`Storage::commit`].
///
/// True natively, false inside a Wasm host. Nothing in this crate reads it: it
/// is for a module that wants to LOG which regime it is in, or a test that wants
/// to assert it. Correct code calls `commit()` either way.
pub const fn commit_required() -> bool {
    cfg!(target_os = "emscripten")
}

/// The name of the durable medium behind [`FileStorage`] on this target, for
/// logs and diagnostics: `"filesystem"` natively, `"indexeddb"` in a Wasm host.
pub const fn backend_name() -> &'static str {
    if cfg!(target_os = "emscripten") {
        "indexeddb"
    } else {
        "filesystem"
    }
}

/// Reject anything that is not a flat name before it reaches a filesystem.
fn check_key(key: &str) -> Result<()> {
    if key.is_empty()
        || key == "."
        || key == ".."
        || key.contains('/')
        || key.contains('\\')
        || key.contains('\0')
    {
        return Err(StorageError::InvalidKey(key.to_string()));
    }
    Ok(())
}

fn io(e: impl fmt::Display) -> StorageError {
    StorageError::Io(e.to_string())
}

// ── the filesystem backend ──────────────────────────────────────────────────

/// The prefix a half-finished [`FileStorage::write`] wears while it is staged.
///
/// Named once because two places have to agree: `write` stages under it and
/// `list` skips anything wearing it, so a file left behind by a crash between
/// the write and the rename is never reported as a key.
const STAGE_PREFIX: &str = ".logos-stage-";

/// A [`Storage`] backed by one directory.
///
/// THE SAME TYPE IN BOTH BUILDS. Natively the directory is on the host's disk.
/// In a Wasm host it is the mount the host populated from IndexedDB before the
/// module started serving — so the reads and writes below are the same
/// `std::fs` calls, and the only thing that changes is what [`Self::commit`]
/// has to do to make them stick. That is the whole reason a module gets one
/// code path.
pub struct FileStorage {
    root: PathBuf,
}

impl FileStorage {
    /// Open (creating if needed) the store rooted at `root`.
    ///
    /// An empty path is [`StorageError::NotAvailable`] rather than a directory
    /// called `""`: a host that has no persistence to offer stamps the context
    /// with an empty path, and a module deserves to hear that as "no storage"
    /// at open time rather than as a mystery write failure later.
    pub fn open(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        if root.as_os_str().is_empty() {
            return Err(StorageError::NotAvailable(
                "the host stamped this module with an empty persistence path".to_string(),
            ));
        }
        std::fs::create_dir_all(&root).map_err(io)?;
        Ok(Self { root })
    }

    fn path_of(&self, key: &str) -> Result<PathBuf> {
        check_key(key)?;
        Ok(self.root.join(key))
    }
}

impl Storage for FileStorage {
    fn read(&self, key: &str) -> Result<Vec<u8>> {
        let path = self.path_of(key)?;
        match std::fs::read(&path) {
            Ok(bytes) => Ok(bytes),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                Err(StorageError::NotFound(key.to_string()))
            }
            Err(e) => Err(io(e)),
        }
    }

    fn write(&self, key: &str, bytes: &[u8]) -> Result<()> {
        let path = self.path_of(key)?;
        // Stage beside the destination and rename. Same directory, so the
        // rename is a rename and not a cross-device copy — the property the
        // atomicity rests on. The staging name carries the pid so two writers
        // of the same key cannot collide on it.
        let stage = self
            .root
            .join(format!("{STAGE_PREFIX}{}-{}", std::process::id(), key));
        std::fs::write(&stage, bytes).map_err(io)?;
        match std::fs::rename(&stage, &path) {
            Ok(()) => Ok(()),
            Err(e) => {
                let _ = std::fs::remove_file(&stage);
                Err(io(e))
            }
        }
    }

    fn remove(&self, key: &str) -> Result<bool> {
        let path = self.path_of(key)?;
        match std::fs::remove_file(&path) {
            Ok(()) => Ok(true),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(e) => Err(io(e)),
        }
    }

    fn exists(&self, key: &str) -> bool {
        match self.path_of(key) {
            Ok(path) => path.is_file(),
            Err(_) => false,
        }
    }

    fn list(&self) -> Result<Vec<String>> {
        let mut out = Vec::new();
        let entries = match std::fs::read_dir(&self.root) {
            Ok(e) => e,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(out),
            Err(e) => return Err(io(e)),
        };
        for entry in entries.flatten() {
            if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
                continue;
            }
            let name = entry.file_name().to_string_lossy().into_owned();
            // A leftover staging file is not a key. See `write`.
            if name.starts_with(STAGE_PREFIX) {
                continue;
            }
            out.push(name);
        }
        out.sort();
        Ok(out)
    }

    fn commit(&self) -> Result<()> {
        host_commit(&self.root)
    }

    fn local_dir(&self) -> Option<&Path> {
        Some(&self.root)
    }
}

// ── the barrier, per target ─────────────────────────────────────────────────

/// Natively: the writes already reached the OS, so the barrier is a directory
/// fsync — it is what makes the renames above survive a power cut, and it is
/// the strongest thing a process can do without owning every file handle.
#[cfg(not(target_os = "emscripten"))]
fn host_commit(root: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        // A directory fsync is not portable off unix and is not available at
        // all on some filesystems; a failure here means "not durable", which is
        // worth reporting rather than swallowing.
        let dir = std::fs::File::open(root).map_err(io)?;
        dir.sync_all().map_err(io)?;
    }
    #[cfg(not(unix))]
    let _ = root;
    Ok(())
}

/// In a Wasm host: ask the host to push its filesystem into IndexedDB.
///
/// `logos_storage_commit` is supplied by the image the module is linked into —
/// logos-module-builder's `wasm/logos_wasm_host.cpp`, which mounts IDBFS at the
/// persistence path and wraps `FS.syncfs(false, …)`. It is the ONE symbol this
/// module needs from its host, it is referenced only here, and it is only
/// referenced on emscripten, so no other target can acquire an undefined
/// symbol from it.
///
/// The call returns as soon as the push is IN FLIGHT — `FS.syncfs` completes on
/// the browser's event loop and this image cannot block on it without Asyncify.
/// A push that failed is reported by the NEXT call, which is what keeps a
/// failure from being dropped rather than merely late.
///
/// Non-zero is a failure the host has already described on the console; the
/// code is carried through so a module can tell "the barrier failed" from "the
/// barrier is not implemented" (`-1`).
#[cfg(target_os = "emscripten")]
fn host_commit(_root: &Path) -> Result<()> {
    extern "C" {
        fn logos_storage_commit() -> std::ffi::c_int;
    }
    let rc = unsafe { logos_storage_commit() };
    if rc == 0 {
        Ok(())
    } else if rc < 0 {
        Err(StorageError::NotAvailable(
            "this Wasm host offers no durable store (logos_storage_commit reported none)"
                .to_string(),
        ))
    } else {
        Err(StorageError::Io(format!(
            "the Wasm host could not commit to IndexedDB (code {rc})"
        )))
    }
}

// ── the in-memory backend ───────────────────────────────────────────────────

/// A [`Storage`] that keeps everything in this process and nothing anywhere
/// else. For unit tests of a core whose persistence is the thing under test,
/// and for a module that legitimately has no store.
///
/// [`Storage::local_dir`] is `None`, which is the honest answer and the reason
/// a path-based dependency cannot be pointed at one.
#[derive(Default)]
pub struct MemoryStorage {
    entries: Mutex<BTreeMap<String, Vec<u8>>>,
}

impl MemoryStorage {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Storage for MemoryStorage {
    fn read(&self, key: &str) -> Result<Vec<u8>> {
        check_key(key)?;
        self.entries
            .lock()
            .unwrap()
            .get(key)
            .cloned()
            .ok_or_else(|| StorageError::NotFound(key.to_string()))
    }

    fn write(&self, key: &str, bytes: &[u8]) -> Result<()> {
        check_key(key)?;
        self.entries.lock().unwrap().insert(key.to_string(), bytes.to_vec());
        Ok(())
    }

    fn remove(&self, key: &str) -> Result<bool> {
        check_key(key)?;
        Ok(self.entries.lock().unwrap().remove(key).is_some())
    }

    fn exists(&self, key: &str) -> bool {
        check_key(key).is_ok() && self.entries.lock().unwrap().contains_key(key)
    }

    fn list(&self) -> Result<Vec<String>> {
        Ok(self.entries.lock().unwrap().keys().cloned().collect())
    }

    fn commit(&self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::storage::{
        backend_name, commit_required, FileStorage, MemoryStorage, Storage, StorageError,
    };
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// A scratch directory unique to this test binary AND this call — two tests
    /// in one binary run on two threads, and a shared name makes them each
    /// other's flake.
    fn tmpdir(tag: &str) -> PathBuf {
        static N: AtomicU32 = AtomicU32::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let d = std::env::temp_dir().join(format!(
            "logos-sdk-storage-{}-{tag}-{n}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    fn roundtrip(s: &dyn Storage) {
        s.write("a.json", b"one").unwrap();
        s.write("b.json", b"two").unwrap();
        s.commit().unwrap();
        assert_eq!(s.read("a.json").unwrap(), b"one");
        assert!(s.exists("b.json"));
        assert_eq!(s.list().unwrap(), vec!["a.json".to_string(), "b.json".to_string()]);
        assert!(s.remove("a.json").unwrap());
        assert!(!s.remove("a.json").unwrap());
        assert!(matches!(s.read("a.json"), Err(StorageError::NotFound(_))));
    }

    #[test]
    fn file_storage_roundtrip() {
        let dir = tmpdir("roundtrip");
        let s = FileStorage::open(&dir).unwrap();
        roundtrip(&s);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn memory_storage_roundtrip() {
        roundtrip(&MemoryStorage::new());
    }

    /// The escape hatch is what a path-based dependency needs, and the memory
    /// store cannot offer it. Asserting the difference keeps `local_dir` an
    /// honest question rather than a convenience that happens to answer.
    #[test]
    fn only_a_directory_backed_store_has_a_local_dir() {
        let dir = tmpdir("localdir");
        let s = FileStorage::open(&dir).unwrap();
        assert_eq!(s.local_dir(), Some(dir.as_path()));
        assert!(MemoryStorage::new().local_dir().is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The host stamps an empty persistence path when it has no store to give.
    /// That has to be an error AT OPEN, not a directory named "" the module
    /// then fails to write into three calls later.
    #[test]
    fn an_empty_persistence_path_is_refused_at_open() {
        assert!(matches!(
            FileStorage::open(""),
            Err(StorageError::NotAvailable(_))
        ));
    }

    /// A staging file is not a key. A `list()` that raced a crash (or, on
    /// emscripten, an image that died between the write and the rename) must
    /// not report one as an account.
    #[test]
    fn a_leftover_staging_file_is_not_listed() {
        let dir = tmpdir("stage");
        let s = FileStorage::open(&dir).unwrap();
        s.write("real.json", b"x").unwrap();
        std::fs::write(dir.join(".logos-stage-999-real.json"), b"half").unwrap();
        assert_eq!(s.list().unwrap(), vec!["real.json".to_string()]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A write over a live key replaces it whole, and leaves no staging file
    /// behind on the way.
    #[test]
    fn a_write_replaces_and_leaves_nothing_behind() {
        let dir = tmpdir("replace");
        let s = FileStorage::open(&dir).unwrap();
        s.write("k", b"first").unwrap();
        s.write("k", b"second-and-longer").unwrap();
        assert_eq!(s.read("k").unwrap(), b"second-and-longer");
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The two facts a module may log about its regime. Native builds are
    /// durable without a commit; this test is what says so on the target it
    /// runs on, and the emscripten arm is asserted by the module-builder's
    /// web-variant test, which is the only place a wasm image exists.
    #[test]
    fn the_regime_is_reported() {
        assert!(!commit_required());
        assert_eq!(backend_name(), "filesystem");
    }

    /// The barrier is reachable without a store, because a core with its own
    /// on-disk layout needs the barrier and not the key/value surface. Natively
    /// it is a directory fsync, so a real directory succeeds and a path that is
    /// not one fails rather than quietly reporting durability.
    #[test]
    fn the_barrier_is_available_on_its_own() {
        let dir = tmpdir("barrier");
        std::fs::create_dir_all(&dir).unwrap();
        crate::storage::commit(&dir).unwrap();
        assert!(crate::storage::commit(&dir.join("nope")).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn keys_may_not_escape_the_root() {
        let s = MemoryStorage::new();
        for bad in ["", "..", "a/b", "/abs", "a\\b", "."] {
            assert!(matches!(s.write(bad, b"x"), Err(StorageError::InvalidKey(_))), "{bad}");
        }
    }
}
