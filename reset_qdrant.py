from config import get_qdrant_client, QDRANT_COLLECTION_NAME

def reset_collection():
    client = get_qdrant_client()
    
    # Lệnh xóa bay màu Collection cũ
    client.delete_collection(collection_name=QDRANT_COLLECTION_NAME)
    print(f"✅ Đã xóa thành công Collection: {QDRANT_COLLECTION_NAME}")

if __name__ == "__main__":
    reset_collection()