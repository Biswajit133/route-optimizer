-- Copilot phase 2b: returning-customer lookups.
-- One row each time the Copilot looks up a phone number's past orders while
-- booking. The widget shows it as the "Returning customer" card, and
-- prepare_order resolves a saved address ("A1") from the latest row of the
-- conversation, so the full address never has to pass through the AI model.
-- New table only; nothing else is changed. Run once per database (dev, then live).

CREATE TABLE copilot_customer_lookups (
    id              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    customer_id     INT NOT NULL,
    user_id         INT NOT NULL,
    conversation_id BIGINT UNSIGNED NOT NULL,
    message_id      BIGINT UNSIGNED NULL,       -- the assistant message that showed the card
    phone           VARCHAR(20) NOT NULL,
    data            MEDIUMTEXT NOT NULL,        -- JSON: card (for the widget), model (what the AI saw), addresses (server only)
    created_at      TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    KEY idx_copilot_lookup_conv (conversation_id, id),
    KEY idx_copilot_lookup_msg (message_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
