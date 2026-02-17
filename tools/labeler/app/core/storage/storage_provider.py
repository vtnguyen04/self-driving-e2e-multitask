from abc import ABC, abstractmethod
import io

class StorageProvider(ABC):
    @abstractmethod
    def save_file(self, content: bytes, filename: str) -> str:
        """Saves file and returns the URI or path."""
        pass

    @abstractmethod
    def get_file_url(self, filename: str) -> str:
        """Returns a URL or path to access the file."""
        pass

    @abstractmethod
    def file_exists(self, filename: str) -> bool:
        """Checks if file exists in storage."""
        pass

class MinioStorageProvider(StorageProvider):
    def __init__(self, endpoint: str, access_key: str, secret_key: str, bucket_name: str, secure: bool = False):
        try:
            from minio import Minio
        except ImportError:
            raise ImportError("The 'minio' library is required for MinioStorageProvider. Install it with 'pip install minio'.")

        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )
        self.bucket_name = bucket_name
        self._ensure_bucket()

    def _ensure_bucket(self):
        if not self.client.bucket_exists(self.bucket_name):
            self.client.make_bucket(self.bucket_name)

    def save_file(self, content: bytes, filename: str) -> str:
        data = io.BytesIO(content)
        self.client.put_object(
            self.bucket_name,
            filename,
            data,
            length=len(content)
        )
        # We return the "minio://bucket/filename" as a virtual path
        return f"minio://{self.bucket_name}/{filename}"

    def get_file_url(self, filename: str) -> str:
        # Generate a temporary presigned URL for the UI to consume
        return self.client.get_presigned_url(
            "GET",
            self.bucket_name,
            filename
        )

    def file_exists(self, filename: str) -> bool:
        try:
            self.client.stat_object(self.bucket_name, filename)
            return True
        except Exception:
            return False

    def get_object(self, filename: str):
        """Returns the raw bytes from MinIO."""
        response = self.client.get_object(self.bucket_name, filename)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()
