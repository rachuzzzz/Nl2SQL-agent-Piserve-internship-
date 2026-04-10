import psycopg2

try:
    conn = psycopg2.connect(
        host="192.168.2.190",
        port=5432,
        database="mydatabase",  # replace with your DB name
        user="myuser",          # replace with your DB username
        password="mypassword"   # replace with your password
    )
    print("✅ Connection successful!")
    conn.close()
except psycopg2.OperationalError as e:
    print("❌ Connection failed!")
    print(e)