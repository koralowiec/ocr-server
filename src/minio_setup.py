import json
import os
from minio import Minio
from minio.error import (
    ResponseError,
    BucketAlreadyOwnedByYou,
    BucketAlreadyExists,
)

minio_host_env = os.environ.get("MINIO_HOST")
minio_host = minio_host_env if minio_host_env is not None else "minio:9000"

minio_access_key = os.environ.get("MINIO_ACCESS_KEY")
minio_secret_key = os.environ.get("MINIO_SECRET_KEY")

minio_client = Minio(
    minio_host,
    access_key=minio_access_key,
    secret_key=minio_secret_key,
    secure=False,
)

bucket_name = "ocr-server"

try:
    minio_client.make_bucket(bucket_name)
except BucketAlreadyOwnedByYou:
    pass
except BucketAlreadyExists:
    pass
except ResponseError as err:
    print(err)
    raise

try:
    # copied from here:
    # https://github.com/minio/minio-py/blob/643dc04b50ba07ad596f7f308e4f20f7a99fabff/examples/set_bucket_policy.py#L61
    policy_read_write = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Action": ["s3:GetBucketLocation"],
                "Sid": "",
                "Resource": [f"arn:aws:s3:::{bucket_name}"],
                "Effect": "Allow",
                "Principal": {"AWS": "*"},
            },
            {
                "Action": ["s3:ListBucket"],
                "Sid": "",
                "Resource": [f"arn:aws:s3:::{bucket_name}"],
                "Effect": "Allow",
                "Principal": {"AWS": "*"},
            },
            {
                "Action": ["s3:ListBucketMultipartUploads"],
                "Sid": "",
                "Resource": [f"arn:aws:s3:::{bucket_name}"],
                "Effect": "Allow",
                "Principal": {"AWS": "*"},
            },
            {
                "Action": [
                    "s3:ListMultipartUploadParts",
                    "s3:GetObject",
                    "s3:AbortMultipartUpload",
                    "s3:DeleteObject",
                    "s3:PutObject",
                ],
                "Sid": "",
                "Resource": [f"arn:aws:s3:::{bucket_name}/*"],
                "Effect": "Allow",
                "Principal": {"AWS": "*"},
            },
        ],
    }
    minio_client.set_bucket_policy(bucket_name, json.dumps(policy_read_write))

except ResponseError as err:
    print(err)
