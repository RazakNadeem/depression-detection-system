import sqlite3

def upgrade_db():
    try:
        conn = sqlite3.connect('depression.db')
        cursor = conn.cursor()
        
        # Check if column exists
        cursor.execute("PRAGMA table_info(medical_report)")
        columns = [info[1] for info in cursor.fetchall()]
        
        if 'patient_id' not in columns:
            print("Adding patient_id column...")
            cursor.execute("ALTER TABLE medical_report ADD COLUMN patient_id TEXT DEFAULT 'Unknown'")
            conn.commit()
            print("Migration successful.")
        else:
            print("Column patient_id already exists.")
            
        conn.close()
    except Exception as e:
        print(f"Migration failed: {e}")

if __name__ == "__main__":
    upgrade_db()
