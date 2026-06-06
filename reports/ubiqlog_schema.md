# UbiqLog Event Schema Reference

**Generated:** 2026-06-06  
**Verified against:** User 1_M (2014 cohort) + User 13_F (2013 cohort) + User 10_M (Iran cohort)

---

## Application Events (PRIMARY)

The `Application` event is the **critical record type** for GraphMind. It represents a single foreground application session with explicit start and end timestamps.

### Fields

| Field         | Type   | Description                              | Example                                  |
|---------------|--------|------------------------------------------|------------------------------------------|
| `ProcessName` | string | Android package ID (app identifier)      | `"com.google.android.gm"`               |
| `Start`       | string | Session start time `MM-D-YYYY HH:MM:SS` | `"11-1-2014 06:00:03"`                  |
| `End`         | string | Session end time `MM-D-YYYY HH:MM:SS`   | `"11-1-2014 06:00:13"`                  |

### Timestamp Format

```
MM-D-YYYY HH:MM:SS
```
- Month: 1–2 digits, no leading zero
- Day: 1–2 digits, no leading zero
- Year: 4 digits
- Hour: 24-hour, no leading zero ambiguity
- Separator: space between date and time, hyphen between date parts

**Parse pattern:** `%m-%d-%Y %H:%M:%S`

### Examples

```json
{"Application": {"ProcessName": "jp.naver.line.android", "Start": "11-1-2014 06:00:03", "End": "11-1-2014 06:00:13"}}
{"Application": {"ProcessName": "com.google.android.gm", "Start": "11-13-2014 23:59:16", "End": "11-13-2014 23:59:26"}}
{"Application": {"ProcessName": "com.android.vending", "Start": "11-13-2014 23:59:26", "End": "11-13-2014 23:59:36"}}
{"Application": {"ProcessName": "com.android.nfc", "Start": "11-13-2014 23:57:56", "End": "11-13-2014 23:59:36"}}
{"Application": {"ProcessName": "com.google.android.deskclock", "Start": "11-14-2014 06:00:01", "End": "11-14-2014 06:00:21"}}
```

### Cardinality Notes

- **Same app, simultaneous sessions:** Common (background services run in parallel)
- **Zero-duration sessions:** `Start == End` — typically background polls, should be filtered or flagged
- **Session duration range:** 0 – 24+ hours
- **Process names include services:** `com.android.nfc`, `com.redbend.vdmc`, `:engine`, `:client` suffixes — filter with known service list
- **Typical apps per user per day:** 10–150 unique process names, 50–500 session records
- **Foreground apps (human-initiated):** Identified by `Start` time gap > 0 seconds and non-system package prefix

---

## WiFi Events

### Fields

| Field          | Type   | Description                          | Example                         |
|----------------|--------|--------------------------------------|---------------------------------|
| `SSID`         | string | Network name                         | `"MyCharterWiFi83-2G"`         |
| `BSSID`        | string | MAC address of access point          | `"c4:04:15:0e:c0:83"`         |
| `capabilities` | string | Security protocol string             | `"[WPA2-PSK-CCMP][WPS][ESS]"` |
| `level`        | string | RSSI signal level (dBm, negative)    | `"-38"`                        |
| `frequency`    | string | Channel frequency in MHz             | `"2462"`                       |
| `time`         | string | Full timestamp (long format)         | `"Saturday, November 1, 2014 12:06:39 AM Pacific Daylight Time"` |

### Timestamp Format (WiFi)

WiFi uses a **different, verbose format** than Application:
```
DayName, Month Day, Year HH:MM:SS AM/PM Timezone
```
**Parse pattern:** `%A, %B %d, %Y %I:%M:%S %p %Z`

### Example

```json
{"WiFi": {"SSID": "MyCharterWiFi83-2G", "BSSID": "c4:04:15:0e:c0:83", "capabilities": "[WPA2-PSK-CCMP][WPS][ESS]", "level": "-38", "frequency": "2462", "time": "Saturday, November 1, 2014 12:01:38 AM Pacific Daylight Time"}}
```

### Notes
- WiFi is by far the **most frequent event type** (~32K–57K records per daily log)
- Multiple SSIDs scanned simultaneously (one JSON line per visible network per scan)
- Signal level in dBm: -38 = strong (nearby), -90 = weak (distant)

---

## Activity Events

### Fields

| Field        | Type   | Description                              | Example                    |
|--------------|--------|------------------------------------------|----------------------------|
| `start`      | string | Activity start `MM-D-YYYY HH:MM:SS`     | `"10-31-2014 19:07:27"`   |
| `end`        | string | Activity end `MM-D-YYYY HH:MM:SS`       | `"11-1-2014 14:19:35"`    |
| `type`       | string | Activity class                           | `"tilting"`, `"walking"`, `"running"`, `"still"` |
| `condfidence`| string | Detection confidence (note: typo in dataset) | `"100"` |

### Examples

```json
{"Activity": {"start": "10-31-2014 19:07:27", "end": "11-1-2014 14:19:35", "type": "tilting", "condfidence": "100"}}
{"Activity": {"start": "11-13-2014 23:19:37", "end": "11-14-2014 07:01:28", "type": "tilting", "condfidence": "100"}}
```

### Notes
- **Typo:** Field name is `condfidence` (not `confidence`) — hardcode this key name
- Activity types observed: `tilting`, `walking`, `running`, `still`, `on_bicycle`, `in_vehicle`
- Activities span multiple days (long-duration records common)

---

## Location Events

### Fields

| Field        | Type   | Description                          | Example              |
|--------------|--------|--------------------------------------|----------------------|
| `Latitude`   | string | GPS latitude (decimal degrees)       | `"35.6913008"`      |
| `Longtitude` | string | GPS longitude (note: typo)           | `"51.374208"`       |
| `Altitude`   | string | Altitude in meters                   | `"0.0"`, `"1670.8"` |
| `time`       | string | Timestamp `MM-D-YYYY HH:MM:SS`      | `"12-31-2013 23:50:08"` |
| `Accuracy`   | string | GPS accuracy radius in meters        | `"28.0"`, `"87.0"`  |
| `Provider`   | string | Location provider                    | `"network"`, `"gps"` |

### Examples

```json
{"Location": {"Latitude": "35.6913008", "Longtitude": "51.374208", "Altitude": "0.0", "time": "12-31-2013 23:50:08", "Accuracy": "28.0", "Provider": "network"}}
{"Location": {"Latitude": "35.76297054", "Longtitude": "51.3383462", "Altitude": "1670.800048828125", "time": "11-11-2013 09:50:57", "Accuracy": "57.0", "Provider": "gps"}}
```

### Notes
- **Typo:** Field name is `Longtitude` (not `Longitude`) — hardcode this key name
- Tehran coordinates (35.7°N, 51.4°E) suggest Iran-based users for the 2013 cohort
- Location NOT present in all users (US users in 2014 cohort: 1_M, 2_F have location)
- Users without Location: no GPS permission or disabled

---

## Bluetooth Events

### Fields

| Field         | Type   | Description               | Example                     |
|---------------|--------|---------------------------|-----------------------------|
| `name`        | string | Device name               | `"NO_DEVICENAME"`, `"iPhone"` |
| `address`     | string | Bluetooth MAC address     | `"2C:B4:3A:08:32:89"`      |
| `bond status` | string | Pairing status            | `"none"`, `"bonded"`        |
| `time`        | string | Verbose timestamp (WiFi format) | `"Saturday, November 1, 2014 2:37:00 PM Pacific Daylight Time"` |

### Example

```json
{"Bluetooth": {"name": "NO_DEVICENAME", "address": "2C:B4:3A:08:32:89", "bond status": "none", "time": "Saturday, November 1, 2014 2:37:00 PM Pacific Daylight Time"}}
```

---

## Call Events

### Fields

| Field      | Type   | Description                   | Example                |
|------------|--------|-------------------------------|------------------------|
| `Number`   | string | Phone number (partially masked)| `"+1805637####"`      |
| `Duration` | string | Call duration in seconds       | `"27"`                |
| `Time`     | string | Call time `MM-D-YYYY HH:MM:SS`| `"11-1-2014 07:48:10"` |
| `Type`     | string | Call type: 1=incoming, 2=outgoing, 3=missed | `"2"` |

### Example

```json
{"Call": {"Number": "+1805637####", "Duration": "27", "Time": "11-1-2014 07:48:10", "Type": "2"}}
```

---

## SMS Events

### Fields

| Field      | Type   | Description                         | Example               |
|------------|--------|-------------------------------------|-----------------------|
| `Address`  | string | Phone number (partially masked)     | `"+98917836####"`    |
| `type`     | string | 1=received, 2=sent                  | `"1"`                |
| `date`     | string | SMS date `MM-D-YYYY HH:MM:SS`      | `"12-13-2013 22:03:49"` |
| `body`     | string | Content (anonymized)                | `"ANONYMIZED"`        |
| `Type`     | string | Duplicate type field (capitalized)  | `"1"`                 |
| `metadata` | object | Optional: contact name              | `{"name": "AzrA"}`  |

### Example

```json
{"SMS": {"Address": "+98917836####", "type": "1", "date": "12-13-2013 22:03:49", "body": "ANONYMIZED", "Type": "1", "metadata": {"name": "AzrA"}}}
```

---

## Timestamp Summary

| Event Type   | Timestamp Field | Format                                          |
|--------------|-----------------|------------------------------------------------|
| Application  | `Start`, `End`  | `MM-D-YYYY HH:MM:SS` (short format)           |
| Activity     | `start`, `end`  | `MM-D-YYYY HH:MM:SS` (short format)           |
| Location     | `time`          | `MM-D-YYYY HH:MM:SS` (short format)           |
| Call         | `Time`          | `MM-D-YYYY HH:MM:SS` (short format)           |
| SMS          | `date`          | `MM-D-YYYY HH:MM:SS` (short format)           |
| WiFi         | `time`          | `DayName, Month D, YYYY H:MM:SS AM/PM TZ`     |
| Bluetooth    | `time`          | `DayName, Month D, YYYY H:MM:SS AM/PM TZ`     |

**Parse pattern for short format:** `%m-%d-%Y %H:%M:%S`

---

## Known Quirks (Implementation Notes)

1. `condfidence` — typo in Activity field name (hardcode)
2. `Longtitude` — typo in Location field name (hardcode)
3. `bond status` — space in Bluetooth field name (use `obj["bond status"]`)
4. WiFi/Bluetooth timestamps are in long locale-aware format (timezone-dependent)
5. Application sessions may overlap (background + foreground at same time)
6. Zero-duration Application sessions exist and need filtering
7. Some filenames use non-ASCII digits (Farsi numerals in user 32_F)
8. Date anomalies: years 2000, 2011 in some files — filter by valid date range
