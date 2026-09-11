# Build report

Built 2026-09-11 05:52. Target: `delay_s` = actual arrival - TransLoc ETA (s).

### Gold (route 29)
- raw rows: 63,566
- detected stop arrivals: 690
- exact duplicate rows dropped: 2,600
- placeholder rows (no stop, no ETA) dropped: 3
- blank-ETA rows dropped (bus at the stop (0-s ETA saved as blank)): 636
- off-route rows (> 50 m from route) dropped: 1,482
- overnight rows (00:00-05:59) dropped: 0
- rows without a known prior stop arrival in the same trajectory dropped: 3,857
- stale-ETA rows (target = the stop just passed) dropped: 526
- rows with the bus already at the target stop (ambiguous ETA) dropped: 485
- rows with no later arrival at the target stop in the same trajectory dropped: 14,105
- labeled rows: 36,218 (dropped 3,654 with actual > 2700s)
- final rows: 36,218; days: ['2026-03-04', '2026-03-05', '2026-03-06']; vehicles: 6
- delay_s: median +266s, mean +327s, p5 -98s, p95 +986s

### Green (route 17)
- raw rows: 96,362
- detected stop arrivals: 799
- exact duplicate rows dropped: 420
- placeholder rows (no stop, no ETA) dropped: 2
- blank-ETA rows dropped (bus at the stop (0-s ETA saved as blank)): 3,312
- off-route rows (> 50 m from route) dropped: 4,132
- overnight rows (00:00-05:59) dropped: 0
- rows without a known prior stop arrival in the same trajectory dropped: 2,099
- stale-ETA rows (target = the stop just passed) dropped: 1,665
- rows with the bus already at the target stop (ambiguous ETA) dropped: 697
- rows with no later arrival at the target stop in the same trajectory dropped: 22,992
- labeled rows: 60,898 (dropped 145 with actual > 2700s)
- final rows: 60,898; days: ['2026-03-09', '2026-03-10', '2026-03-16']; vehicles: 6
- delay_s: median +285s, mean +335s, p5 -28s, p95 +893s

### Red (route 20)
- raw rows: 30,974
- detected stop arrivals: 3,958
- exact duplicate rows dropped: 0
- placeholder rows (no stop, no ETA) dropped: 3
- blank-ETA rows dropped (bus off its route): 2,191
- off-route rows (> 50 m from route) dropped: 505
- overnight rows (00:00-05:59) dropped: 1,568
- rows without a known prior stop arrival in the same trajectory dropped: 44
- **causal next-stop check on Red**: 'stop after the last known arrival' = TransLoc's stop for 75.6% of 11,504 rows with the bus between stops
- stale-ETA rows (target = the stop just passed) dropped: 11,823
- rows with the bus already at the target stop (ambiguous ETA) dropped: 729
- rows with no later arrival at the target stop in the same trajectory dropped: 644
- labeled rows: 13,453 (dropped 14 with actual > 2700s)
- final rows: 13,453; days: ['2026-03-03', '2026-03-04', '2026-03-05']; vehicles: 9
- delay_s: median +7s, mean +64s, p5 -35s, p95 +330s

### Clough (route 28)
- raw rows: 12,076
- detected stop arrivals: 2,376
- exact duplicate rows dropped: 0
- placeholder rows (no stop, no ETA) dropped: 0
- blank-ETA rows dropped (bus at the stop (0-s ETA saved as blank)): 0
- off-route rows (> 50 m from route) dropped: 139
- overnight rows (00:00-05:59) dropped: 0
- rows without a known prior stop arrival in the same trajectory dropped: 97
- rows with the bus dwelling at a stop (ambiguous next stop) dropped: 6,098
- stale-ETA rows (target = the stop just passed) dropped: 0
- rows with the bus already at the target stop (ambiguous ETA) dropped: 0
- rows with no later arrival at the target stop in the same trajectory dropped: 207
- labeled rows: 5,518 (dropped 17 with actual > 2700s)
- final rows: 5,518; days: ['2026-03-02', '2026-03-03', '2026-03-04', '2026-03-05', '2026-03-06']; vehicles: 12
- delay_s: median +20s, mean +178s, p5 -25s, p95 +880s

**Clough excluded from combined: causal next-stop agreement on Red only 75.6% (< 80%).**

### dataset `combined`
  embeddings {'vehicle_id': 4, 'route_stop_id': 4} trained 11 epochs (val MSE std-units 1.311)
- rows 110,569 (train 81,428 / test 29,141); cv groups 7 (test 2); features 54
- train days ['2026-03-03', '2026-03-05', '2026-03-06', '2026-03-10', '2026-03-16']; test days ['2026-03-04', '2026-03-09']; vehicles: 18; stops: 49

### dataset `route`
  embeddings {'vehicle_id': 2, 'route_stop_id': 4} trained 23 epochs (val MSE std-units 1.978)
- rows 60,898 (train 43,900 / test 16,998); cv groups 3 (test 1); features 47
- train days ['2026-03-10', '2026-03-16']; test days ['2026-03-09']; vehicles: 6; stops: 10

Chosen bus for `bus` dataset: Green vehicle 3 (23,992 rows; days ['2026-03-09', '2026-03-10'])

### dataset `bus`
  embeddings {'route_stop_id': 4} trained 9 epochs (val MSE std-units 0.209)
- rows 23,992 (train 16,456 / test 7,536); cv groups 8 (test 3); features 41
- train days ['2026-03-10']; test days ['2026-03-09']; vehicles: 1; stops: 10
