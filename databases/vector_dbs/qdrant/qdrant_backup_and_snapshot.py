"""
Qdrant Backup & Snapshot Utilities
----------------------------------
Create, list, download, and restore Qdrant snapshots.

Dependencies:
    pip install qdrant-client requests
"""

from qdrant_client import QdrantClient
import requests
import os
from datetime import datetime

# ==========================================
# 1️⃣ Connect to Qdrant
# ==========================================
client = QdrantClient(host="localhost", port=6333)
print("✅ Connected to Qdrant for backup operations")

# ==========================================
# 2️⃣ Create snapshot
# ==========================================
def create_collection_snapshot(collection_name: str):
    """Creates a snapshot of a specific collection."""
    snapshot_info = client.create_snapshot(collection_name)
    print(f"📦 Snapshot created for '{collection_name}': {snapshot_info.name}")
    return snapshot_info


def create_full_snapshot():
    """Creates a snapshot of the entire Qdrant database."""
    snapshot_info = client.create_full_snapshot()
    print(f"🗄️ Full DB snapshot created: {snapshot_info.name}")
    return snapshot_info


# ==========================================
# 3️⃣ List existing snapshots
# ==========================================
def list_snapshots(collection_name: str = None):
    """List all available snapshots."""
    if collection_name:
        snapshots = client.list_snapshots(collection_name)
    else:
        snapshots = client.list_full_snapshots()

    print("📋 Available snapshots:")
    for s in snapshots:
        print(f" - {s.name} ({s.size / 1024 / 1024:.2f} MB)")
    return snapshots


# ==========================================
# 4️⃣ Download snapshot
# ==========================================
def download_snapshot(snapshot_name: str, save_path="backups/", collection_name: str = None):
    """Download snapshot from Qdrant to local storage."""
    os.makedirs(save_path, exist_ok=True)

    if collection_name:
        url = f"http://localhost:6333/collections/{collection_name}/snapshots/{snapshot_name}/download"
    else:
        url = f"http://localhost:6333/snapshots/{snapshot_name}/download"

    print(f"⬇️  Downloading snapshot from {url}")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        local_file = os.path.join(save_path, snapshot_name)
        with open(local_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ Snapshot saved to {local_file}")
    else:
        print(f"❌ Failed to download snapshot: {response.status_code} - {response.text}")


# ==========================================
# 5️⃣ Restore snapshot
# ==========================================
def restore_snapshot(snapshot_name: str, collection_name: str = None):
    """
    Restore snapshot from local or remote file system.
    Note: Must be placed in Qdrant’s configured snapshot directory before calling restore.
    """
    if collection_name:
        result = client.recover_snapshot(collection_name, snapshot_name)
        print(f"🧩 Restored collection '{collection_name}' from snapshot {snapshot_name}")
    else:
        result = client.recover_full_snapshot(snapshot_name)
        print(f"🧩 Restored full database from snapshot {snapshot_name}")
    return result


# ==========================================
# 6️⃣ Example workflow
# ==========================================
if __name__ == "__main__":
    COLLECTION_NAME = "ai_snippets"

    # Step 1: Create snapshot
    snapshot = create_collection_snapshot(COLLECTION_NAME)

    # Step 2: List snapshots
    list_snapshots(COLLECTION_NAME)

    # Step 3: Download the snapshot
    download_snapshot(snapshot.name, save_path="backups/", collection_name=COLLECTION_NAME)

    # Optional Step 4: Restore
    # restore_snapshot(snapshot.name, collection_name=COLLECTION_NAME)
