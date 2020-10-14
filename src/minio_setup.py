import json
from minio import Minio
from minio.error import (
    ResponseError,
    BucketAlreadyOwnedByYou,
    BucketAlreadyExists,
)

minioClient = Minio(
    "minio:9000",
    access_key="AKIAIOSFODNN7EXAMPLE",
    secret_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    secure=False,
)

bucket_name = "ocr-server"

try:
    minioClient.make_bucket(bucket_name)
except BucketAlreadyOwnedByYou:
    pass
except BucketAlreadyExists:
    pass
except ResponseError as err:
    print(err)
    raise

try:
    # copied from here: https://github.com/minio/minio-py/blob/643dc04b50ba07ad596f7f308e4f20f7a99fabff/examples/set_bucket_policy.py#L61
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
    minioClient.set_bucket_policy(bucket_name, json.dumps(policy_read_write))

except ResponseError as err:
    print(err)
