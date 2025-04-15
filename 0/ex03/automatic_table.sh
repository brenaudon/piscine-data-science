#!/bin/bash

CSV_DIR="subject/customer"
OUTPUT_SQL="automatic_table.sql"
DB_USER="brenaudo"
DB_NAME="piscineds"
DB_HOST="localhost"

# Clear previous SQL content
> "$OUTPUT_SQL"

# Loop through CSV files and generate SQL
for csv_file in "$CSV_DIR"/*.csv; do
table_name=$(basename "$csv_file" .csv)

cat <<EOF >> "$OUTPUT_SQL"
CREATE TABLE IF NOT EXISTS $table_name (
    event_time TIMESTAMPTZ,
event_type TEXT,
product_id INTEGER,
price NUMERIC(10, 2),
user_id BIGINT,
user_session UUID
);

EOF

    # Properly format the \COPY command on a single line without semicolon
    echo "\\COPY $table_name(event_time, event_type, product_id, price, user_id, user_session) FROM '$csv_file' DELIMITER ',' CSV HEADER" >> "$OUTPUT_SQL"
    echo "" >> "$OUTPUT_SQL"

done

# Execute the generated SQL script
psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -f "$OUTPUT_SQL"