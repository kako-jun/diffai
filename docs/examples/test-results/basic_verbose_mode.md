# diffai Test: basic/verbose_mode

**Description:** Verbose mode showing detailed processing information

**Command:** `diffai tests/fixtures/config_v1.json tests/fixtures/config_v2.json --verbose`

**Generated:** Mon Jul 14 02:00:32 PM JST 2025
**Version:** v0.3.4

## Command Output

```
=== diffai verbose mode enabled ===
Configuration:
  Input format: None
  Output format: Cli
  Recursive mode: false

File analysis:
  Input 1: tests/fixtures/config_v1.json
  Input 2: tests/fixtures/config_v2.json
  Detected format: Json
  File 1 size: 128 bytes
  File 2 size: 129 bytes

Processing results:
  Total processing time: 146.000µs
  Differences found: 2
  Format-specific analysis: Json

    ~ app.settings.log_level: "info" -> "debug"
  ~ app.version: "1.0" -> "1.1"
```

**Exit Code:** 0

✅ **Status:** SUCCESS

---
