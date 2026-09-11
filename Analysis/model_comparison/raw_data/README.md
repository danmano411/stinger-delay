# Raw data

Byte-identical copies of the scraped files already in this repo, plus two cached API responses. Nothing here is edited; all cleaning happens in `../src/build_datasets.py`.

| Folder | Files | Source in repo | Route | Dates (2026) | Notes |
|---|---|---|---|---|---|
| `gold/eta_scraper/` | 3 | `Analysis/bus_data/` | Gold (29) | Mar 4–6 | 21 rows per bus per poll (ETA to every stop), 30 s polling. Scraped before the Mar 16 fix, so an ETA of 0 s is saved as blank |
| `gold/gps_only_scraper/` | 1 | `Analysis/alina_polling/route_29_data.csv` | Gold (29) | Mar 9–13 | Stop and ETA columns 100% empty (the scraper read the wrong JSON keys). **Not used for modeling** (no target) |
| `green/` | 4 | `Analysis/green_bus_data/` | Green (17) | Mar 9, 10, 11, 16 | 10 rows per bus per poll, 15 s polling; 0-s ETA saved as blank; the Mar 11 file is a single snapshot |
| `red/` | 1 | `Platform/red_line/data/red_line_data.csv` | Red (20) | Mar 3–6 | Next-stop ETA only, UTC timestamps, 24/7 collector. Blank ETA = bus off its route |
| `clough/` | 1 | `Analysis/clough_bus/clough_bus_data.csv` | Clough (28) | Mar 2–6 | Next-stop ETA only; stop ID 100% empty (scraper read `RouteStopId` instead of `RouteStopID`); ETA stored as `/Date(ms)/` (UTC). **Not used for modeling**: which stop an ETA refers to can't be recovered without looking at the future (see the main README) |
| `route_config/` | 1 | live `GetRoutesForMapWithSchedule` (gatech.transloc.com), fetched 2026-09-10 | all | — | Stop coordinates, stop order, planned `SecondsToNextStop` / `SecondsAtStop`, route shapes. Every stop ID in the March data is present |
| `weather/` | 1 | Open-Meteo historical archive API, GT coordinates (33.7756, −84.3963) | — | Mar 2–17 (UTC) | Hourly °F / mph / inch, in **UTC**. The build converts to America/New_York with real DST rules; the API's local-time option applies one fixed offset to the whole range, which would shift Mar 2–7 by an hour |

Weather request:
`https://archive-api.open-meteo.com/v1/archive?latitude=33.7756&longitude=-84.3963&start_date=2026-03-02&end_date=2026-03-17&hourly=temperature_2m,apparent_temperature,relative_humidity_2m,precipitation,rain,cloud_cover,wind_speed_10m,wind_gusts_10m,weather_code&temperature_unit=fahrenheit&wind_speed_unit=mph&precipitation_unit=inch&timezone=GMT`
