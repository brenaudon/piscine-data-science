BEGIN;

SELECT 'Before removal' AS info, COUNT(*) AS row_count
FROM customers;

-- Create a deduplicated table using a window function

CREATE TABLE customers_dedup AS
WITH cte AS (
    SELECT
        c.*,
        LAG(event_time) OVER (
            PARTITION BY event_type, product_id, price, user_id, user_session
            ORDER BY event_time
        ) AS prev_event_time
    FROM customers c
)
SELECT
    event_time,
    event_type,
    product_id,
    price,
    user_id,
    user_session
FROM cte
WHERE
   -- Keep the row if there's no previous row in the partition
    prev_event_time IS NULL
   -- OR if the difference is strictly greater than 1 second
   OR (event_time - prev_event_time) > INTERVAL '1 second';

-- Drop the old customers table
DROP TABLE customers;

-- Rename the deduplicated table to "customers"
ALTER TABLE customers_dedup RENAME TO customers;

SELECT 'After removal' AS info, COUNT(*) AS row_count
FROM customers;

COMMIT;
