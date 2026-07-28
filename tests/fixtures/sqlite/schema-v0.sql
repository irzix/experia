-- Deterministic Experia SQLite fixture: supported legacy schema version 0.
PRAGMA foreign_keys = ON;
BEGIN TRANSACTION;

CREATE TABLE experiences (
    id TEXT PRIMARY KEY,
    task TEXT NOT NULL,
    action TEXT NOT NULL,
    result TEXT NOT NULL,
    context TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE lessons (
    id TEXT PRIMARY KEY,
    experience_id TEXT NOT NULL,
    content TEXT NOT NULL,
    confidence REAL NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (experience_id) REFERENCES experiences (id)
);

CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    type TEXT NOT NULL,
    confidence REAL NOT NULL,
    importance REAL NOT NULL,
    source TEXT,
    metadata TEXT,
    embedding TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    expires_at TEXT
);

CREATE INDEX idx_exp_created ON experiences(created_at DESC);
CREATE INDEX idx_lessons_exp ON lessons(experience_id);

INSERT INTO experiences (
    id, task, action, result, context, created_at
) VALUES (
    '11111111-1111-4111-8111-111111111111',
    'Migrate nested agent state',
    'Load a deterministic legacy fixture',
    'Legacy records remained readable',
    '{"attempt":2,"flags":[true,false,null],"request":{"limits":{"retries":3,"timeout_seconds":1.5},"locale":"en-CA"},"tags":["migration","nested"]}',
    '2024-01-15T08:30:45.123456+05:30'
);

INSERT INTO lessons (
    id, experience_id, content, confidence, created_at
) VALUES (
    '22222222-2222-4222-8222-222222222222',
    '11111111-1111-4111-8111-111111111111',
    'Preserve source values before rebuilding derived data.',
    0.875,
    '2024-01-15T04:15:00-04:00'
);

INSERT INTO memories (
    id, content, type, confidence, importance, source, metadata, embedding,
    created_at, updated_at, expires_at
) VALUES (
    '33333333-3333-4333-8333-333333333333',
    'Prefer deterministic migrations — preserve café metadata.',
    'strategy',
    0.925,
    0.975,
    '11111111-1111-4111-8111-111111111111',
    '{"audit":{"actors":["agent","reviewer"],"verified":true},"labels":["nested","fixture"],"metrics":{"loss":0.125,"samples":3}}',
    '[0.125,-0.5,1.25,0.0]',
    '2024-01-16T12:00:00+09:00',
    '2024-01-16T03:30:00+00:00',
    '2025-06-01T18:45:30-07:00'
), (
    '44444444-4444-4444-8444-444444444444',
    'Retain ordered embeddings during schema upgrades.',
    'lesson',
    0.75,
    0.625,
    '22222222-2222-4222-8222-222222222222',
    '{"constraints":{"regions":["us-east","eu-west"],"retry":{"backoff":[0.25,0.5],"maximum":2}},"nullable":null}',
    '[-1.0,0.75,0.5]',
    '2024-02-29T23:59:59+14:00',
    '2024-02-29T05:59:59-04:00',
    NULL
);

PRAGMA user_version = 0;
COMMIT;
