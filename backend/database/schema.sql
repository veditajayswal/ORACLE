CREATE TABLE IF NOT EXISTS assets (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL DEFAULT 'motor',
    location TEXT DEFAULT 'Default Facility',
    sensor_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'HEALTHY',
    health REAL DEFAULT 100.0
);

CREATE TABLE IF NOT EXISTS telemetry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id TEXT NOT NULL,
    timestamp REAL NOT NULL,
    accel_x REAL NOT NULL,
    accel_y REAL NOT NULL,
    accel_z REAL NOT NULL,
    vibration REAL NOT NULL,
    temperature REAL,
    rpm REAL,
    FOREIGN KEY(asset_id) REFERENCES assets(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id TEXT NOT NULL,
    timestamp REAL NOT NULL,
    health REAL NOT NULL,
    anomaly_score REAL NOT NULL,
    failure_risk REAL NOT NULL,
    trend TEXT DEFAULT 'STABLE',
    explanation TEXT,
    model_version TEXT DEFAULT 'v1.0',
    FOREIGN KEY(asset_id) REFERENCES assets(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS maintenance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id TEXT NOT NULL,
    timestamp REAL NOT NULL,
    issue TEXT NOT NULL,
    action TEXT NOT NULL,
    component TEXT NOT NULL,
    result TEXT NOT NULL,
    FOREIGN KEY(asset_id) REFERENCES assets(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id TEXT NOT NULL,
    timestamp REAL NOT NULL,
    event_type TEXT NOT NULL,
    description TEXT NOT NULL,
    severity TEXT DEFAULT 'INFO',
    features TEXT,
    FOREIGN KEY(asset_id) REFERENCES assets(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS baselines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id TEXT NOT NULL,
    feature TEXT NOT NULL,
    mean REAL NOT NULL,
    std REAL NOT NULL,
    created_at REAL NOT NULL,
    version TEXT DEFAULT 'v1.0',
    FOREIGN KEY(asset_id) REFERENCES assets(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS commands (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id TEXT NOT NULL,
    timestamp REAL NOT NULL,
    command TEXT NOT NULL,
    status TEXT DEFAULT 'PENDING',
    source TEXT DEFAULT 'DECISION_ENGINE',
    FOREIGN KEY(asset_id) REFERENCES assets(id) ON DELETE CASCADE
);
