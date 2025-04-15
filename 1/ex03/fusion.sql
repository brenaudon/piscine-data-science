BEGIN;

-- Create a new table with all columns from customers plus the item columns
-- LEFT JOIN so we keep all rows from customers even if items is missing.
CREATE TABLE customers_fused AS
SELECT
    c.*,
    i.category_id,
    i.category_code,
    i.brand
FROM customers c
         LEFT JOIN items i ON c.product_id = i.product_id;

-- Drop the old customers table
DROP TABLE customers;

-- Rename the fused table to customers
ALTER TABLE customers_fused RENAME TO customers;

COMMIT;