import sqlite3

def check_detections_schema():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(detections)")
    columns = cursor.fetchall()
    print("Detections Schema:")
    for col in columns:
        print(col)
    conn.close()

if __name__ == "__main__":
    check_detections_schema()
