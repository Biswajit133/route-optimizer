-- Pathoconnect Copilot - order actions from chat (reschedule, cancel, assign phlebo, payment link).
-- Run manually on the `hsc` database (dev first, then live), after 001-003.
-- One new table, no change to any existing table.
--
-- The model never changes an order. Its prepare_* tools only write a row here
-- (status 'pending'); the CRM user clicks the button on the card, and
-- POST /copilot/actions/<id>/confirm calls the same orders-service API the CRM
-- uses (rescheduleOrder, cancelOrder, assign_phlebo, send-paymentlink-customer)
-- with the user's own JWT.

CREATE TABLE copilot_action_drafts (
    id              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    customer_id     INT NOT NULL,
    user_id         INT NOT NULL,
    conversation_id BIGINT UNSIGNED NOT NULL,
    message_id      BIGINT UNSIGNED NULL,          -- assistant message that showed the card
    action          VARCHAR(20) NOT NULL,          -- reschedule | cancel | assign | payment_link
    payload         MEDIUMTEXT NOT NULL,           -- JSON: what is sent to the orders API
    summary         MEDIUMTEXT NOT NULL,           -- JSON: what the card shows
    status          VARCHAR(20) NOT NULL DEFAULT 'pending', -- pending | running | done | partial | failed | cancelled | replaced
    result          MEDIUMTEXT NULL,               -- JSON: per-row outcome, payment link ...
    error           VARCHAR(500) NULL,
    expires_at      DATETIME NOT NULL,             -- UTC
    confirmed_at    DATETIME NULL,
    created_at      TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP NULL DEFAULT NULL,
    PRIMARY KEY (id),
    KEY idx_copilot_action_conv (conversation_id, status),
    KEY idx_copilot_action_msg (message_id),
    KEY idx_copilot_action_customer (customer_id, created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
