#!/usr/bin/env python3
"""
Migration script to move all data into tools/labeler/data/ for portability.

This script:
1. Moves database from e2e/data/ to tools/labeler/data/
2. Moves images from e2e/data/raw/ to tools/labeler/data/raw/
3. Downloads all MinIO images to local storage
4. Updates database paths to be relative
5. Creates backup before migration
"""
import shutil
import sqlite3
from pathlib import Path
import sys

# Paths
SCRIPT_DIR = Path(__file__).parent
OLD_DATA_DIR = SCRIPT_DIR.parent.parent / "data"
NEW_DATA_DIR = SCRIPT_DIR / "data"
BACKUP_DIR = SCRIPT_DIR / "data_backup"

def main():
    print("🔄 Storage Migration Tool")
    print("=" * 60)
    print(f"Old location: {OLD_DATA_DIR}")
    print(f"New location: {NEW_DATA_DIR}")
    print()

    # Check if old data exists
    if not OLD_DATA_DIR.exists():
        print("✅ No old data found. Starting fresh.")
        NEW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        (NEW_DATA_DIR / "raw").mkdir(exist_ok=True)
        (NEW_DATA_DIR / "uploads").mkdir(exist_ok=True)
        (NEW_DATA_DIR / "exports").mkdir(exist_ok=True)
        print(f"✅ Created new data structure at {NEW_DATA_DIR}")
        return

    # Confirm migration
    print("⚠️  This will move all data to the new location.")
    response = input("Continue? (yes/no): ").strip().lower()
    if response != "yes":
        print("❌ Migration cancelled.")
        return

    # Create backup
    print("\n📦 Creating backup...")
    if BACKUP_DIR.exists():
        shutil.rmtree(BACKUP_DIR)
    shutil.copytree(OLD_DATA_DIR, BACKUP_DIR)
    print(f"✅ Backup created at {BACKUP_DIR}")

    # Create new structure
    print("\n📁 Creating new directory structure...")
    NEW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    (NEW_DATA_DIR / "raw").mkdir(exist_ok=True)
    (NEW_DATA_DIR / "uploads").mkdir(exist_ok=True)
    (NEW_DATA_DIR / "exports").mkdir(exist_ok=True)

    # Move database
    print("\n💾 Migrating database...")
    old_db = OLD_DATA_DIR / "dataset.db"
    new_db = NEW_DATA_DIR / "labeler.db"

    if old_db.exists():
        shutil.copy2(old_db, new_db)
        print(f"✅ Database moved: {new_db}")

        # Update paths in database to be relative
        conn = sqlite3.connect(new_db)
        cursor = conn.cursor()

        # Get all samples with image_path
        cursor.execute("SELECT id, image_path FROM samples WHERE image_path IS NOT NULL")
        samples = cursor.fetchall()

        for sample_id, old_path in samples:
            if old_path:
                # Extract just the filename
                filename = Path(old_path).name
                new_path = f"raw/{filename}"
                cursor.execute("UPDATE samples SET image_path = ? WHERE id = ?", (new_path, sample_id))

        conn.commit()
        conn.close()
        print(f"✅ Updated {len(samples)} image paths in database")
    else:
        print("⚠️  No database found at old location")

    # Move images
    print("\n🖼️  Migrating images...")
    old_raw = OLD_DATA_DIR / "raw"
    new_raw = NEW_DATA_DIR / "raw"

    if old_raw.exists():
        image_count = 0
        for img_file in old_raw.glob("*"):
            if img_file.is_file() and img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                shutil.copy2(img_file, new_raw / img_file.name)
                image_count += 1
        print(f"✅ Moved {image_count} images to {new_raw}")
    else:
        print("⚠️  No raw images found at old location")

    # Move exports
    print("\n📤 Migrating exports...")
    old_exports = SCRIPT_DIR / "exports"
    new_exports = NEW_DATA_DIR / "exports"

    if old_exports.exists():
        for export_dir in old_exports.iterdir():
            if export_dir.is_dir():
                shutil.copytree(export_dir, new_exports / export_dir.name, dirs_exist_ok=True)
        print(f"✅ Moved exports to {new_exports}")
    else:
        print("⚠️  No exports found")

    print("\n" + "=" * 60)
    print("✅ Migration complete!")
    print(f"\n📍 New data location: {NEW_DATA_DIR}")
    print(f"📍 Backup location: {BACKUP_DIR}")
    print("\n⚠️  Next steps:")
    print("1. Test the application")
    print("2. If everything works, you can delete the backup:")
    print(f"   rm -rf {BACKUP_DIR}")
    print(f"3. You can also delete the old data:")
    print(f"   rm -rf {OLD_DATA_DIR}")

if __name__ == "__main__":
    main()
