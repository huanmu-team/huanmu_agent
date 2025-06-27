import os


ZILLIZ_BASE_URL = os.environ.get(
    "ZILLIZ_BASE_URL",
    "https://in03-c5553ad7d167d0b.serverless.gcp-us-west1.cloud.zilliz.com"
)
ZILLIZ_TOKEN = os.environ.get(
    "ZILLIZ_API_KEY",
    "2f0c801fc87a680f01e0ef97690be2c43170dab70d591623edb04f4c0af119ceb237c561591c076c20f35657e67f570a829cdc0d"
)

GOOGLE_AI_STUDIO_API_KEY = os.environ.get(
    "GOOGLE_AI_STUDIO_API_KEY",
    "AIzaSyA7385mpgBJTNrJCl3a_ndtdJHnHFyKfE8"
)