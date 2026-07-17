//! Crate-level constants for the ai-gateway.
//!
//! Per `litellm-rust/CLAUDE.md`, magic numbers and fixed strings live here
//! (the Rust mirror of Python's `litellm/constants.py`), not inline in feature
//! modules. Env-overridable tunables keep their `DEFAULT_*` value here; the env
//! read + fallback happens at the host/config layer.

/// Default total polling deadline for Azure Document Intelligence analyze
/// operations, in seconds.
/// Override: `AZURE_OPERATION_POLLING_TIMEOUT`.
pub(crate) const DEFAULT_AZURE_OPERATION_POLLING_TIMEOUT_SECS: u64 = 120;

/// Env var name for the Azure Document Intelligence polling deadline override.
pub(crate) const AZURE_OPERATION_POLLING_TIMEOUT_ENV: &str = "AZURE_OPERATION_POLLING_TIMEOUT";

/// Default LiteLLM control-plane base URL for request-log egress when
/// `LITELLM_PROXY_BASE_URL` is unset.
#[cfg(feature = "server")]
pub(crate) const DEFAULT_PROXY_BASE_URL: &str = "http://localhost:4000";

/// The logs ingest path appended to the proxy base. Not a tunable; it is the
/// proxy's API contract (the rust-control-plane router on the Python proxy).
#[cfg(feature = "server")]
pub(crate) const RUST_CONTROL_PLANE_LOGS_PATH: &str = "/v1/rust_control_plane/logs";

/// Default bounded channel depth for the log-egress worker.
/// Override: `LITELLM_LOG_CHANNEL_CAPACITY`.
#[cfg(feature = "server")]
pub(crate) const DEFAULT_CHANNEL_CAPACITY: usize = 4096;

/// Default max records POSTed per request to the control plane.
/// Override: `LITELLM_LOG_BATCH_SIZE`.
#[cfg(feature = "server")]
pub(crate) const DEFAULT_MAX_BATCH_SIZE: usize = 256;

/// Default partial-batch flush cadence, in ms.
/// Override: `LITELLM_LOG_FLUSH_INTERVAL_MS`.
#[cfg(feature = "server")]
pub(crate) const DEFAULT_FLUSH_INTERVAL_MS: u64 = 500;

/// Provider attributed to realtime sessions in the logging payload.
#[cfg(feature = "server")]
pub(crate) const DEFAULT_PROVIDER: &str = "openai";
