# ORACLE — Smart Intelligence for Physical Machines

ORACLE turns ordinary machines into self-learning digital twins. A low-cost sensor node learns a machine's normal motion and vibration "fingerprint," remembers its history, detects early degradation, predicts failure risk, and explains what to do about it — all visualized live through a 3D/AR Android app.

## The Problem

Machine failures are usually only noticed after performance has already degraded or a breakdown has happened, leading to expensive downtime and unplanned repairs. Most existing monitoring relies on fixed thresholds (e.g. "if vibration > X, alert") applied the same way regardless of what the machine is actually doing. That approach misses gradual degradation and produces false alarms, because the same vibration pattern can be perfectly normal under one operating condition (e.g. full load) and abnormal under another (e.g. idle).

## The Idea

Instead of one fixed threshold per machine, ORACLE:

- Learns what "normal" looks like for that specific machine, separately for each operating condition (startup, steady load, shutdown, etc.)
- Remembers the machine's full history — past readings, past anomalies, maintenance performed, past predictions, and outcomes
- Detects drift away from the learned baseline as an early warning sign, before failure occurs
- Predicts failure risk with a trend and confidence level, not just a binary "fault detected"
- Explains its reasoning: what changed → why it may be happening → what could happen next → what action to take
- Recommends a concrete action: continue operating, schedule maintenance, or stop the machine
- Represents all of this as a live 3D digital twin in an Android app, with AR overlay and the ability to trigger a physical response when risk is critical

We start with a small motor/pump as our demo machine, but the same system is designed to generalize to fans, compressors, vehicles, lab equipment, and other physical assets — giving every physical asset its own digital identity, history, health score, and predictions.

## How It Works

```
Physical Machine → Sensors → Data → Learn Normal Behavior → Detect Changes → Predict Problems → Recommend Action
```

**1. Sensing**
A small sensor node is attached to the machine:
- ESP32 microcontroller
- Motion/accelerometer sensor
- Vibration/acoustic sensor

The sensors continuously capture how the machine moves and vibrates and stream that data off the device.

**2. Learning normal behavior**
Rather than hardcoding thresholds, ORACLE first learns the machine's own normal fingerprint from real operating data, ideally broken down by operating condition (speed, load, startup/shutdown) rather than a single fixed baseline.

**3. Detecting change**
Live readings are continuously compared against the learned baseline for the current operating condition. Meaningful drift is flagged as a potential anomaly — changes in behavior, unusual vibration/movement, gradual degradation, or patterns that historically preceded failure.

**4. Machine memory**
Every machine keeps its own history: previous sensor readings, detected abnormalities, maintenance performed, previous predictions, and what actually happened afterward. This lets ORACLE compare current condition against the machine's own past rather than a generic standard.

**5. Prediction and explanation**
When behavior starts changing, ORACLE estimates a risk level with a trend and confidence (e.g. "high possibility this machine is developing a problem") and explains the reasoning behind it, rather than issuing an opaque "fault detected" flag.

**6. Decision support**
Each risk assessment maps to one of three concrete recommendations: continue operating, schedule maintenance, or stop the machine — along with the supporting evidence.

**7. Android app + digital twin**
The system is accessible through an Android app that lets a user:
1. Scan/register a physical machine using the phone camera
2. Create its digital profile
3. Connect the sensor device
4. View live machine data
5. View health and history
6. View detected problems
7. View predicted future problems

The machine is represented as an interactive 3D model inside the app, with live health info attached to individual parts (e.g. `Motor → Health: 82%`, `Bearing → Possible degradation`, `Sensor → Connected/Working`). The 3D model is the interface for understanding the physical machine, not just a visual — AR can overlay this information directly onto the real machine, and critical risk levels can trigger a physical response through the ESP32 (e.g. an alert, shutoff relay).

## Example Output

```
⚠ Pump health: 64%
- Vibration increasing
- Behavior different from its normal baseline
- Similar pattern appeared before previous maintenance
- Failure risk increasing

Recommended: Inspect the machine soon.
```

## Training Data

We don't rely on a pre-existing dataset — data is generated directly from the demo hardware:

- **Normal data**: run the sensor on the healthy demo motor/pump and record it as the baseline.
- **Fault data**: deliberately introduce faults (e.g. unbalance the shaft, loosen a mount, restrict airflow, alter voltage/speed) to generate abnormal examples to test detection against.
- **Public datasets (optional/supplementary)**: datasets like CWRU or MAFAULDA (motor/bearing vibration data with normal and fault conditions) can supplement testing, though they won't exactly match our own sensor/machine setup.
- For a prototype, a full trained ML model isn't required — a statistical baseline (mean + normal spread of readings during healthy operation, flagged when live readings deviate significantly) is sufficient to demonstrate the concept convincingly.

## Tech Stack

| Layer | Components |
|---|---|
| Hardware | ESP32, accelerometer, vibration/acoustic sensor |
| Data pipeline | Sensor → WiFi/Bluetooth → processing service |
| Modeling | Per-machine baseline learning, drift/anomaly detection, risk scoring |
| Storage | Machine history log (readings, anomalies, maintenance, predictions, outcomes) |
| App | Android app with live monitoring, health/history views, 3D digital twin, AR overlay |
| Actuation | Physical trigger (alert/shutoff) via ESP32 on critical risk |

## Build Roadmap

1. Set up hardware — wire ESP32 + accelerometer + vibration sensor to the demo motor/pump
2. Collect raw sensor data while the machine runs normally (ideally across a few operating conditions)
3. Stream sensor data reliably to a processing service (phone/laptop/server)
4. Learn a normal baseline from the collected data
5. Detect drift/anomalies by comparing live data to the baseline
6. Add machine memory (log of readings, anomalies, maintenance, predictions, outcomes)
7. Convert anomalies into a risk score/trend with an explanation
8. Convert risk into a recommendation (continue / schedule maintenance / stop)
9. Build the Android app shell showing live data and health status
10. Add the 3D digital twin with live part-level health info
11. Add AR overlay (stretch goal)
12. Add physical trigger action on critical risk
13. Rehearse the live demo: normal running → introduced fault → detection → risk explanation → recommendation → physical trigger
14. Polish explanations so every alert shows "what changed → why → what's next → what to do," not just a number

## Why This Matters Beyond the Demo

The pump is only the first demonstration. The same system can extend to motors, fans, compressors, vehicles, laboratory equipment, and other industrial machines. The core idea is giving every physical asset its own digital identity, history, health, and predictions — with the hardware kept intentionally simple. The real innovation is the software layer that turns raw sensor data into an actual understanding of the machine's condition.