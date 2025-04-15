DO $$
DECLARE
tbl record;
BEGIN
    -- Drop the customers table if it exists already
EXECUTE 'DROP TABLE IF EXISTS customers CASCADE';

-- Create an empty 'customers' table
EXECUTE 'CREATE TABLE customers AS TABLE data_2022_oct WITH NO DATA';

-- Loop over all tables that match data_202% in public schema
FOR tbl IN
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'public'
  AND table_name LIKE 'data_202%'
    LOOP
        RAISE NOTICE 'Inserting from table %', tbl.table_name;

-- Insert all rows from that table into customers
EXECUTE format('INSERT INTO customers SELECT * FROM %I', tbl.table_name);
END LOOP;

END $$;
