# UbiqLog Dataset Structure Report

**Generated:** 2026-06-06  
**Source:** UbiqLog — UCI Repository (Smartphone Lifelogging Dataset)  
**Location:** `datasets/ubiqlog/UbiqLog4UCI/`

---

## Directory Tree

```
datasets/ubiqlog/
├── UbiqLog4UCI/                    ← Primary dataset root
│   ├── 1_M/                        ← User 1, Male
│   │   ├── log_11-1-2014.txt       ← Daily log (6.7 MB)
│   │   ├── log_11-2-2014.txt
│   │   ├── ... (30 daily files)
│   │   └── log_11-30-2014.txt      ← Last day (3.6 MB)
│   ├── 2_F/                        ← User 2, Female
│   ├── 3_M/
│   ├── ...
│   └── 35_F/                       ← User 35, Female
└── __MACOSX/                       ← macOS artifact (ignored)
```

---

## User Directory Summary

| User   | Files | Size (MB) | First Log            | Last Log             | Gender |
|--------|-------|-----------|----------------------|----------------------|--------|
| 1_M    | 30    | 230.0     | log_11-1-2014.txt    | log_11-30-2014.txt   | M      |
| 2_F    | 27    | 87.1      | log_11-1-2014.txt    | log_11-27-2014.txt   | F      |
| 3_M    | —     | —         | —                    | —                    | M      |
| 4_F    | —     | —         | —                    | —                    | F      |
| 5_F    | —     | —         | —                    | —                    | F      |
| 6_M    | —     | —         | —                    | —                    | M      |
| 7_F    | —     | —         | —                    | —                    | F      |
| 8_M    | —     | —         | —                    | —                    | M      |
| 9_M    | —     | —         | —                    | —                    | M      |
| 10_M   | 9     | 0.2       | log_11-29-2013.txt   | log_12-7-2013.txt    | M      |
| 11_F   | 61    | 26.4      | log_1-1-2014.txt     | log_12-9-2013.txt    | F      |
| 12_M   | 56    | 16.2      | log_1-10-2014.txt    | log_12-9-2013.txt    | M      |
| 13_F   | 52    | 152.5     | log_1-2-2014.txt     | log_12-7-2013.txt    | F      |
| 14_F   | 70    | 3.8       | log_1-1-2014.txt     | log_12-9-2013.txt    | F      |
| 15_F   | 24    | 3.7       | log_11-24-2013.txt   | log_12-7-2013.txt    | F      |
| 16_F   | 55    | 67.4      | log_1-10-2014.txt    | log_12-9-2013.txt    | F      |
| 17_F   | 57    | 13.0      | log_1-1-2014.txt     | log_12-9-2013.txt    | F      |
| 18_F   | 65    | 76.3      | log_1-1-2014.txt     | log_12-9-2013.txt    | F      |
| 19_F   | 57    | 106.6     | log_1-1-2014.txt     | log_12-9-2013.txt    | F      |
| 20_M   | 69    | 3.6       | log_1-1-2014.txt     | log_12-9-2013.txt    | M      |
| 21_F   | 35    | 61.9      | log_1-10-2014.txt    | log_12-6-2013.txt    | F      |
| 22_M   | 49    | 12.1      | log_1-1-2014.txt     | log_12-9-2013.txt    | M      |
| 23_F   | 28    | 39.3      | log_1-1-2000.txt     | log_12-9-2013.txt    | F      |
| 24_F   | 51    | 150.7     | log_1-1-2014.txt     | log_12-9-2013.txt    | F      |
| 25_F   | 8     | 0.1       | log_11-22-2013.txt   | log_11-29-2013.txt   | F      |
| 26_F   | 48    | 2.7       | log_1-1-2014.txt     | log_12-9-2013.txt    | F      |
| 27_F   | 55    | 15.8      | log_1-1-2014.txt     | log_12-9-2013.txt    | F      |
| 28_F   | 73    | 112.1     | log_1-1-2014.txt     | log_12-9-2013.txt    | F      |
| 29_F   | 57    | 18.6      | log_1-1-2014.txt     | log_12-9-2013.txt    | F      |
| 30_F   | 30    | 7.0       | log_1-1-2014.txt     | log_12-31-2013.txt   | F      |
| 31_F   | 67    | 28.8      | log_1-1-2011.txt     | log_12-9-2013.txt    | F      |
| 32_F   | 42    | 5.9       | —                    | —                    | F      |
| 33_F   | —     | —         | —                    | —                    | F      |
| 34_F   | —     | —         | —                    | —                    | F      |
| 35_F   | —     | —         | —                    | —                    | F      |

> Note: Full per-user statistics computed in Phase 2 user analysis script.

---

## File Format

- **Format:** Plain text, one JSON object per line (NDJSON)
- **Filename pattern:** `log_MM-DD-YYYY.txt`
- **Encoding:** UTF-8 (some users use non-Latin locales — Farsi, Arabic digits in filenames)
- **Delimiter:** Newline-separated JSON objects (not arrays)
- **File size range:** 100 KB – 12 MB per daily log

---

## Inferred Hierarchy

```
Dataset
  └── User (ID_Gender, e.g. "1_M", "13_F")
        └── Day (log_MM-DD-YYYY.txt)
              └── Event (JSON line)
                    ├── Application  ← PRIMARY: app usage sessions
                    ├── WiFi         ← Context: nearby networks
                    ├── Bluetooth    ← Context: nearby BT devices
                    ├── Activity     ← Context: physical activity
                    ├── Call         ← Context: phone calls
                    ├── SMS          ← Context: text messages
                    └── Location     ← Context: GPS/network position
```

---

## File Count Summary

| Metric               | Value          |
|----------------------|----------------|
| Total users          | 35             |
| Total log files      | ~1,340+        |
| Total dataset size   | ~1.4 GB        |
| Date range           | 2011–2014      |
| Primary collection   | Nov 2013–Jan 2014 |
| Files per user range | 8–73           |
| Max file size        | 12.3 MB        |

---

## Anomalies Noted

1. **User 23_F:** Has a file `log_1-1-2000.txt` — clearly a clock error on the device
2. **User 31_F:** Has a file `log_1-1-2011.txt` — another clock anomaly
3. **User 32_F:** Filenames contain Farsi/Arabic numeral digits (Unicode) — filename parsing requires encoding-aware handling
4. **User 10_M:** Only 9 files, 0.2 MB — very sparse data
5. **User 25_F:** Only 8 files, 0.1 MB — very sparse data
6. **User 1_M and 2_F:** November 2014 — different cohort from most others (Nov/Dec 2013)

> Anomalous date files (2000, 2011) will be filtered by timestamp parsing, not filename.
