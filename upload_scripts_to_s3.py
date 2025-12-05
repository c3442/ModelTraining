import boto3
import os

BUCKET_NAME = "8up-model-training"
S3_PREFIX = "script"  # folder inside the bucket

FILES_TO_UPLOAD = [
    "fine_tune_gemma.py",
    "requirements.txt",
]


def main():
    s3 = boto3.client("s3")

    for file_path in FILES_TO_UPLOAD:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        s3_key = f"{S3_PREFIX}/{os.path.basename(file_path)}"
        print(f"Uploading {file_path} → s3://{BUCKET_NAME}/{s3_key}")

        s3.upload_file(file_path, BUCKET_NAME, s3_key)

    print("✅ Upload completed successfully.")

if __name__ == "__main__":
    main()
