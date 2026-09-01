//! Detection of the memory available to this process, used to auto-size the
//! default (fair) memory pool the way SedonaDB does.
//!
//! The probe prefers the cgroup limit over the physical RAM size so that a
//! containerized deployment (e.g. a 12 GiB cgroup on a 64 GiB node) sizes its
//! pool from the memory it is actually allowed to use:
//!
//! 1. cgroup v2: `/sys/fs/cgroup/memory.max` (the literal `max` means no limit)
//! 2. cgroup v1: `/sys/fs/cgroup/memory/memory.limit_in_bytes` (a huge
//!    sentinel value close to `i64::MAX` means no limit)
//! 3. total system RAM (`/proc/meminfo` on Linux, `sysctl hw.memsize` on macOS)

use std::path::Path;

const CGROUP_V2_MEMORY_MAX: &str = "/sys/fs/cgroup/memory.max";
const CGROUP_V1_MEMORY_LIMIT: &str = "/sys/fs/cgroup/memory/memory.limit_in_bytes";

/// cgroup v1 reports "no limit" as a huge number (`i64::MAX` rounded down to
/// the page size, e.g. 9223372036854771712). Treat anything at or above 1 EiB
/// as "no limit"; no real container is given an exbibyte of memory.
const CGROUP_NO_LIMIT_SENTINEL: u64 = 1 << 60;

/// The memory available to this process in bytes: the cgroup limit if the
/// process runs under one, otherwise the total system RAM.
/// Returns `None` if nothing can be detected.
pub fn available_memory() -> Option<u64> {
    read_cgroup_limit(Path::new(CGROUP_V2_MEMORY_MAX))
        .or_else(|| read_cgroup_limit(Path::new(CGROUP_V1_MEMORY_LIMIT)))
        .or_else(total_system_memory)
}

/// Parse a cgroup memory limit file (either version).
/// Returns `None` if the file is missing, unparsable, or expresses "no limit"
/// (the literal `max` for cgroup v2, a huge sentinel for cgroup v1).
fn read_cgroup_limit(path: &Path) -> Option<u64> {
    let content = std::fs::read_to_string(path).ok()?;
    parse_cgroup_limit(&content)
}

fn parse_cgroup_limit(content: &str) -> Option<u64> {
    let content = content.trim();
    if content == "max" {
        return None;
    }
    let limit: u64 = content.parse().ok()?;
    if limit == 0 || limit >= CGROUP_NO_LIMIT_SENTINEL {
        return None;
    }
    Some(limit)
}

#[cfg(target_os = "linux")]
fn total_system_memory() -> Option<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    parse_meminfo_total(&meminfo)
}

#[cfg(target_os = "linux")]
fn parse_meminfo_total(meminfo: &str) -> Option<u64> {
    // The `MemTotal` line looks like: `MemTotal:       16384000 kB`
    let line = meminfo.lines().find(|x| x.starts_with("MemTotal:"))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
}

#[cfg(target_os = "macos")]
fn total_system_memory() -> Option<u64> {
    let mut size: u64 = 0;
    let mut len = std::mem::size_of::<u64>();
    // SAFETY: `hw.memsize` is a 64-bit integer sysctl; we pass a pointer to a
    // `u64` and its length, and check the return code.
    let ret = unsafe {
        libc::sysctlbyname(
            c"hw.memsize".as_ptr(),
            &mut size as *mut u64 as *mut libc::c_void,
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if ret == 0 && size > 0 {
        Some(size)
    } else {
        None
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn total_system_memory() -> Option<u64> {
    None
}

#[cfg(test)]
#[expect(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn write_file(dir: &Path, name: &str, content: &str) -> std::path::PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, content).unwrap();
        path
    }

    #[test]
    fn test_parse_cgroup_limit() {
        // cgroup v2 with a limit
        assert_eq!(parse_cgroup_limit("12884901888\n"), Some(12884901888));
        // cgroup v2 without a limit
        assert_eq!(parse_cgroup_limit("max\n"), None);
        // cgroup v1 without a limit (i64::MAX rounded down to the page size)
        assert_eq!(parse_cgroup_limit("9223372036854771712\n"), None);
        // garbage
        assert_eq!(parse_cgroup_limit(""), None);
        assert_eq!(parse_cgroup_limit("unlimited"), None);
        // zero is not a usable limit
        assert_eq!(parse_cgroup_limit("0"), None);
    }

    #[test]
    fn test_read_cgroup_limit_fall_through() {
        let dir = tempfile::tempdir().unwrap();

        // A v2-style file with a real limit wins.
        let v2 = write_file(dir.path(), "memory.max", "1073741824\n");
        assert_eq!(read_cgroup_limit(&v2), Some(1073741824));

        // A v2-style file with "max" falls through to the next source.
        let v2_max = write_file(dir.path(), "memory.max.unlimited", "max\n");
        let v1 = write_file(dir.path(), "memory.limit_in_bytes", "2147483648\n");
        let detected = read_cgroup_limit(&v2_max).or_else(|| read_cgroup_limit(&v1));
        assert_eq!(detected, Some(2147483648));

        // A missing file falls through as well.
        let missing = dir.path().join("does-not-exist");
        let detected = read_cgroup_limit(&missing).or_else(|| read_cgroup_limit(&v1));
        assert_eq!(detected, Some(2147483648));

        // v1 sentinel is treated as absent.
        let v1_unlimited = write_file(
            dir.path(),
            "memory.limit_in_bytes.unlimited",
            "9223372036854771712\n",
        );
        assert_eq!(read_cgroup_limit(&v1_unlimited), None);
    }

    #[test]
    fn test_total_system_memory() {
        // On the supported development and deployment platforms the probe
        // must find the physical memory size, and it must be sane (at least
        // 64 MiB, no more than 64 TiB).
        let total = total_system_memory().expect("total system memory should be detectable");
        assert!(total >= 64 * 1024 * 1024, "total memory too small: {total}");
        assert!(
            total <= 64 * 1024 * 1024 * 1024 * 1024,
            "total memory too large: {total}"
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_parse_meminfo_total() {
        let meminfo = "MemTotal:       16384000 kB\nMemFree:         1234567 kB\n";
        assert_eq!(parse_meminfo_total(meminfo), Some(16384000 * 1024));
        assert_eq!(parse_meminfo_total("MemFree: 1 kB\n"), None);
    }
}
