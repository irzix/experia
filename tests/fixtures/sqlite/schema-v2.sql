-- Deterministic Experia SQLite fixture: supported current schema version 2.
PRAGMA foreign_keys = ON;
BEGIN TRANSACTION;

CREATE TABLE experiences (
    id TEXT PRIMARY KEY,
    task TEXT NOT NULL,
    action TEXT NOT NULL,
    result TEXT NOT NULL,
    agent_role TEXT NOT NULL DEFAULT 'default',
    context TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE lessons (
    id TEXT PRIMARY KEY,
    experience_id TEXT NOT NULL,
    content TEXT NOT NULL,
    agent_role TEXT NOT NULL DEFAULT 'default',
    root_cause TEXT,
    confidence REAL NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (experience_id) REFERENCES experiences (id)
);

CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    type TEXT NOT NULL,
    agent_role TEXT NOT NULL DEFAULT 'default',
    confidence REAL NOT NULL,
    importance REAL NOT NULL,
    source TEXT,
    metadata TEXT,
    embedding TEXT,
    reinforcement_count INTEGER NOT NULL DEFAULT 0,
    success_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    expires_at TEXT,
    embedding_dimension INTEGER
        CHECK (embedding_dimension IS NULL OR embedding_dimension >= 0)
);

CREATE TABLE schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    applied_at TEXT NOT NULL
);

CREATE TABLE memory_vector_bands (
    memory_id TEXT NOT NULL,
    dimension INTEGER NOT NULL,
    band INTEGER NOT NULL,
    bucket INTEGER NOT NULL,
    index_version INTEGER NOT NULL,
    PRIMARY KEY (memory_id, band),
    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
);

CREATE INDEX idx_exp_created ON experiences(created_at DESC);
CREATE INDEX idx_lessons_exp ON lessons(experience_id);
CREATE INDEX idx_mem_role_type ON memories(agent_role, type);
CREATE INDEX idx_mem_rank ON memories(importance DESC, confidence DESC);
CREATE INDEX idx_mem_expires ON memories(expires_at);
CREATE INDEX idx_vector_bucket
    ON memory_vector_bands(dimension, index_version, band, bucket);

INSERT INTO experiences (
    id, task, action, result, agent_role, context, created_at
) VALUES (
    '11111111-1111-4111-8111-111111111111',
    'Migrate nested agent state',
    'Load a deterministic legacy fixture',
    'Legacy records remained readable',
    'migration-specialist',
    '{"attempt":2,"flags":[true,false,null],"request":{"limits":{"retries":3,"timeout_seconds":1.5},"locale":"en-CA"},"tags":["migration","nested"]}',
    '2024-01-15T08:30:45.123456+05:30'
);

INSERT INTO lessons (
    id, experience_id, content, agent_role, root_cause, confidence, created_at
) VALUES (
    '22222222-2222-4222-8222-222222222222',
    '11111111-1111-4111-8111-111111111111',
    'Preserve source values before rebuilding derived data.',
    'migration-specialist',
    'A previous migration path lacked an explicit version registry.',
    0.875,
    '2024-01-15T04:15:00-04:00'
);

INSERT INTO memories (
    id, content, type, agent_role, confidence, importance, source, metadata,
    embedding, reinforcement_count, success_count, created_at, updated_at,
    expires_at, embedding_dimension
) VALUES (
    '33333333-3333-4333-8333-333333333333',
    'Prefer deterministic migrations — preserve café metadata.',
    'strategy',
    'global',
    0.925,
    0.975,
    '11111111-1111-4111-8111-111111111111',
    '{"audit":{"actors":["agent","reviewer"],"verified":true},"labels":["nested","fixture"],"metrics":{"loss":0.125,"samples":3}}',
    '[0.125,-0.5,1.25,0.0]',
    7,
    5,
    '2024-01-16T12:00:00+09:00',
    '2024-01-16T03:30:00+00:00',
    '2025-06-01T18:45:30-07:00',
    4
), (
    '44444444-4444-4444-8444-444444444444',
    'Retain ordered embeddings during schema upgrades.',
    'lesson',
    'analysis-agent',
    0.75,
    0.625,
    '22222222-2222-4222-8222-222222222222',
    '{"constraints":{"regions":["us-east","eu-west"],"retry":{"backoff":[0.25,0.5],"maximum":2}},"nullable":null}',
    '[-1.0,0.75,0.5]',
    2,
    1,
    '2024-02-29T23:59:59+14:00',
    '2024-02-29T05:59:59-04:00',
    NULL,
    3
);

INSERT INTO memory_vector_bands (
    memory_id, dimension, band, bucket, index_version
) VALUES
    ('33333333-3333-4333-8333-333333333333', 4, 0, 1204, 1),
    ('33333333-3333-4333-8333-333333333333', 4, 1, -77, 1),
    ('44444444-4444-4444-8444-444444444444', 3, 0, 803, 1),
    ('44444444-4444-4444-8444-444444444444', 3, 1, 42, 1);

INSERT INTO schema_migrations (version, name, applied_at) VALUES
    (1, 'normalize_source_schema', '2024-01-01T00:00:00+00:00'),
    (2, 'add_vector_index_schema', '2024-01-02T00:00:00+00:00');

PRAGMA user_version = 2;
COMMIT;
