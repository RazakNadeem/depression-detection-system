from app import app, db
import os

db_path = os.path.join("instance", "depression.db")
if os.path.exists(db_path):
    os.remove(db_path)
    print(f"Deleted old database file at {db_path}.")

with app.app_context():
    db.create_all()
    print("Created new database schema.")
