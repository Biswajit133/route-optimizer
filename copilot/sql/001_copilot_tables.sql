-- Pathoconnect Copilot - phase 1 tables.
-- Run manually on the `hsc` database (dev first, then live). Nothing in the
-- app creates these. All new tables; the only change to an existing table is
-- one new column on `customers` (default 0 = Copilot off for every customer).

ALTER TABLE customers
    ADD COLUMN copilot_feature TINYINT(1) NOT NULL DEFAULT 0;

-- LLM settings. The row with customer_id NULL is the global default; a row
-- with a customer_id overrides it for that customer, and every NULL field in
-- that row means "use the global value". Edited from the HSC console
-- (/hsc-console-a0ada463/copilot); the Python service caches it for 60s.
-- api_key_enc is AES-256-GCM, "v1:" + base64(iv[12] + tag[16] + ciphertext),
-- with the shared COPILOT_ENC_KEY (customers service + Python service .env).
CREATE TABLE copilot_llm_settings (
    id                  INT UNSIGNED NOT NULL AUTO_INCREMENT,
    customer_id         INT NULL,
    provider            VARCHAR(30) NULL,          -- groq | openai | anthropic | openai_compatible
    base_url            VARCHAR(255) NULL,         -- blank = the provider's default URL
    model               VARCHAR(120) NULL,
    api_key_enc         TEXT NULL,
    api_key_last4       VARCHAR(8) NULL,           -- shown in the console only
    temperature         DECIMAL(3,2) NULL,
    max_tokens          INT NULL,
    daily_message_limit INT NULL,                  -- per customer per day (user messages)
    is_active           TINYINT(1) NOT NULL DEFAULT 1,
    updated_by          VARCHAR(100) NULL,
    created_at          TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP NULL DEFAULT NULL,
    PRIMARY KEY (id),
    KEY idx_copilot_llm_settings_customer (customer_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

INSERT INTO copilot_llm_settings
    (customer_id, provider, base_url, model, temperature, max_tokens, daily_message_limit, is_active, updated_by)
VALUES
    (NULL, 'groq', NULL, 'openai/gpt-oss-120b', 0.20, 1024, 200, 1, 'setup-sql');

CREATE TABLE copilot_conversations (
    id          BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    customer_id INT NOT NULL,
    user_id     INT NOT NULL,
    title       VARCHAR(160) NULL,
    created_at  TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMP NULL DEFAULT NULL,
    deleted_at  TIMESTAMP NULL DEFAULT NULL,
    PRIMARY KEY (id),
    KEY idx_copilot_conv_user (customer_id, user_id, updated_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- One row per chat turn side. Token counts and provider/model sit on the
-- assistant row; the daily limit counts role = 'user' rows.
CREATE TABLE copilot_messages (
    id                BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    conversation_id   BIGINT UNSIGNED NOT NULL,
    customer_id       INT NOT NULL,
    user_id           INT NOT NULL,
    role              ENUM('user','assistant') NOT NULL,
    content           MEDIUMTEXT NULL,
    provider          VARCHAR(30) NULL,
    model             VARCHAR(120) NULL,
    prompt_tokens     INT NULL,
    completion_tokens INT NULL,
    latency_ms        INT NULL,
    error             VARCHAR(500) NULL,
    created_at        TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    KEY idx_copilot_msg_conv (conversation_id, id),
    KEY idx_copilot_msg_usage (customer_id, created_at, role)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Audit: every tool the model ran, with its arguments.
CREATE TABLE copilot_tool_calls (
    id              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    message_id      BIGINT UNSIGNED NULL,           -- the assistant message it fed
    conversation_id BIGINT UNSIGNED NOT NULL,
    customer_id     INT NOT NULL,
    user_id         INT NOT NULL,
    tool            VARCHAR(60) NOT NULL,
    arguments       TEXT NULL,
    ok              TINYINT(1) NOT NULL DEFAULT 1,
    error           VARCHAR(500) NULL,
    duration_ms     INT NULL,
    created_at      TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    KEY idx_copilot_tool_conv (conversation_id),
    KEY idx_copilot_tool_customer (customer_id, created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Optional but recommended: a read-only DB user for the Copilot's data
-- queries (set COPILOT_READ_DB_USER / COPILOT_READ_DB_PASSWORD). The copilot_*
-- tables above are written with the normal DB_USER.
-- CREATE USER 'copilot_ro'@'%' IDENTIFIED BY '<strong password>';
-- GRANT SELECT ON hsc.* TO 'copilot_ro'@'%';
