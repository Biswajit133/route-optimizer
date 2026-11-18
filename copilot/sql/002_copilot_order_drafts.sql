-- Pathoconnect Copilot - phase 2 (order create from chat).
-- Run manually on the `hsc` database (dev first, then live), after 001.
-- One new table, no change to any existing table.
--
-- The model never creates an order. Its prepare_order tool only writes a row
-- here (status 'pending'); the CRM user clicks Confirm on the card, and
-- POST /copilot/drafts/<id>/confirm sends `payload` to the orders service
-- create-order API with the user's own JWT.

CREATE TABLE copilot_order_drafts (
    id              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    customer_id     INT NOT NULL,
    user_id         INT NOT NULL,
    conversation_id BIGINT UNSIGNED NOT NULL,
    message_id      BIGINT UNSIGNED NULL,          -- assistant message that showed the card
    payload         MEDIUMTEXT NOT NULL,           -- JSON: create-order form fields
    summary         MEDIUMTEXT NOT NULL,           -- JSON: what the card shows
    status          VARCHAR(20) NOT NULL DEFAULT 'pending', -- pending | creating | created | failed | cancelled | replaced
    order_id        INT NULL,
    visit_id        INT NULL,
    order_number    VARCHAR(60) NULL,
    error           VARCHAR(500) NULL,
    expires_at      DATETIME NOT NULL,
    confirmed_at    DATETIME NULL,
    created_at      TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP NULL DEFAULT NULL,
    PRIMARY KEY (id),
    KEY idx_copilot_draft_conv (conversation_id, status),
    KEY idx_copilot_draft_msg (message_id),
    KEY idx_copilot_draft_customer (customer_id, created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
