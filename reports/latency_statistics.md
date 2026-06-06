# Galaxy A23 App Launch Latency Statistics

**Source:** `datasets/app_launch_latency.csv`  
**Device:** Samsung Galaxy A23 (SM-A235F/DS, Android 14, OneUI 6.1)  
**Measurement:** ADB `am start -W`, 100 launches per app per tier  
**Total rows:** 3,900 (13 apps × 3 tiers × 100 samples)  

---

## Cold Start Latency (ms)

| App | Package | Category | Mean | Median | P90 | P95 | P99 | Std |
|-----|---------|----------|------|--------|-----|-----|-----|-----|
| netflix | `com.netflix.mediaclient` | media | 4292 | 4285 | 4940 | 5346 | 5613 | 542 |
| youtube | `com.google.android.youtube` | media | 3564 | 3634 | 4078 | 4262 | 4604 | 439 |
| slack | `com.Slack` | productivity | 3430 | 3433 | 3935 | 4164 | 4509 | 419 |
| instagram | `com.instagram.android` | social | 3268 | 3320 | 3725 | 3860 | 3940 | 397 |
| google_maps | `com.google.android.apps.maps` | utility | 3174 | 3130 | 3667 | 3842 | 4010 | 369 |
| amazon | `com.amazon.mShop.android.shopping` | commerce | 2945 | 2946 | 3377 | 3514 | 3688 | 362 |
| spotify | `com.spotify.music` | media | 2849 | 2801 | 3350 | 3469 | 3616 | 334 |
| samsung_health | `com.sec.android.app.shealth` | health | 2557 | 2539 | 2954 | 2988 | 3139 | 284 |
| paytm | `net.one97.paytm` | financial | 2364 | 2378 | 2794 | 2849 | 3006 | 323 |
| whatsapp | `com.whatsapp` | social | 2256 | 2286 | 2591 | 2668 | 2773 | 276 |
| phonepe | `com.phonepe.app` | financial | 2104 | 2108 | 2428 | 2517 | 2678 | 271 |
| gmail | `com.google.android.gm` | productivity | 1828 | 1836 | 2141 | 2162 | 2386 | 242 |
| chrome | `com.android.chrome` | utility | 1285 | 1298 | 1528 | 1573 | 1706 | 200 |

## Warm Start Latency (ms)

| App | Mean | Median | P90 | P95 | P99 | Std | Saved vs Cold |
|-----|------|--------|-----|-----|-----|-----|---------------|
| netflix | 2040 | 2042 | 2272 | 2347 | 2430 | 200 | **2253 ms** |
| youtube | 1697 | 1710 | 1948 | 1988 | 2169 | 197 | **1867 ms** |
| slack | 1603 | 1572 | 1883 | 1921 | 1978 | 184 | **1827 ms** |
| instagram | 1539 | 1538 | 1768 | 1798 | 1928 | 180 | **1729 ms** |
| google_maps | 1480 | 1482 | 1666 | 1735 | 1814 | 159 | **1693 ms** |
| amazon | 1385 | 1394 | 1557 | 1599 | 1638 | 140 | **1560 ms** |
| spotify | 1378 | 1400 | 1587 | 1636 | 1718 | 162 | **1471 ms** |
| samsung_health | 1203 | 1206 | 1381 | 1416 | 1478 | 134 | **1355 ms** |
| paytm | 1145 | 1145 | 1297 | 1335 | 1392 | 123 | **1220 ms** |
| whatsapp | 993 | 996 | 1157 | 1186 | 1208 | 123 | **1263 ms** |
| phonepe | 987 | 976 | 1142 | 1176 | 1274 | 108 | **1117 ms** |
| gmail | 860 | 854 | 995 | 1034 | 1082 | 105 | **968 ms** |
| chrome | 605 | 603 | 716 | 750 | 813 | 95 | **680 ms** |

## Hot Start Latency (ms)

| App | Mean | Median | P90 | P95 | P99 | Std | Saved vs Cold |
|-----|------|--------|-----|-----|-----|-----|---------------|
| netflix | 496 | 490 | 614 | 651 | 666 | 84 | **3797 ms** |
| youtube | 408 | 404 | 498 | 542 | 571 | 73 | **3156 ms** |
| slack | 366 | 362 | 461 | 467 | 490 | 64 | **3064 ms** |
| instagram | 326 | 327 | 384 | 403 | 428 | 53 | **2942 ms** |
| google_maps | 293 | 300 | 362 | 391 | 419 | 59 | **2880 ms** |
| amazon | 280 | 276 | 364 | 386 | 409 | 65 | **2665 ms** |
| spotify | 276 | 277 | 339 | 357 | 389 | 49 | **2572 ms** |
| samsung_health | 239 | 238 | 313 | 340 | 363 | 55 | **2319 ms** |
| paytm | 215 | 217 | 262 | 269 | 290 | 38 | **2150 ms** |
| phonepe | 193 | 194 | 241 | 254 | 285 | 37 | **1911 ms** |
| whatsapp | 191 | 192 | 237 | 247 | 268 | 39 | **2065 ms** |
| gmail | 162 | 162 | 212 | 224 | 237 | 38 | **1666 ms** |
| chrome | 113 | 116 | 149 | 155 | 161 | 28 | **1172 ms** |

## Latency Savings Summary

| App | Cold (ms) | Warm (ms) | Hot (ms) | Warm Saves | Hot Saves | Hot Saves % |
|-----|-----------|-----------|----------|-----------|-----------|-------------|
| netflix | 4292 | 2040 | 496 | 2253 ms | 3797 ms | 88.4% |
| youtube | 3564 | 1697 | 408 | 1867 ms | 3156 ms | 88.6% |
| slack | 3430 | 1603 | 366 | 1827 ms | 3064 ms | 89.3% |
| instagram | 3268 | 1539 | 326 | 1729 ms | 2942 ms | 90.0% |
| google_maps | 3174 | 1480 | 293 | 1693 ms | 2880 ms | 90.8% |
| amazon | 2945 | 1385 | 280 | 1560 ms | 2665 ms | 90.5% |
| spotify | 2849 | 1378 | 276 | 1471 ms | 2572 ms | 90.3% |
| samsung_health | 2557 | 1203 | 239 | 1355 ms | 2319 ms | 90.7% |
| paytm | 2364 | 1145 | 215 | 1220 ms | 2150 ms | 90.9% |
| whatsapp | 2256 | 993 | 191 | 1263 ms | 2065 ms | 91.5% |
| phonepe | 2104 | 987 | 193 | 1117 ms | 1911 ms | 90.8% |
| gmail | 1828 | 860 | 162 | 968 ms | 1666 ms | 91.2% |
| chrome | 1285 | 605 | 113 | 680 ms | 1172 ms | 91.2% |

**Average cold start:** 2763 ms  
**Average warm start:** 1301 ms  
**Average hot start:** 274 ms  
**Average warm saving:** 1462 ms (52.9%)  
**Average hot saving:** 2489 ms (90.1%)  
