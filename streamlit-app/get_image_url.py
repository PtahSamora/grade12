# from azure.storage.blob import BlobServiceClient, BlobClient
import io
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions, BlobClient
from vision_analysis import vision_analysis
from datetime import datetime, timedelta
def generate_blob_sas_url(blob_name):
            account_name = 'mathappanalysis'
            account_key = 'your_account_key'
            container_name = 'mathapp-container'

            sas_blob = generate_blob_sas(
                account_name=account_name,
                container_name=container_name,
                blob_name=blob_name,
                account_key=account_key,
                permission=BlobSasPermissions(read=True),
                expiry=datetime.utcnow() + timedelta(hours=1)  # Link is valid for 1 hour
            )
            # img_url = 'https://mathappanalysis.blob.core.windows.net/mathapp-container/test_image.jpg?sp=r&st=2024-05-12T14:38:57Z&se=2024-05-16T22:38:57Z&spr=https&sv=2022-11-02&sr=b&sig=r%2Fcv0KG291rk6b2tml5kUB1PhG9jvRa1JNi%2Brn7IrXY%3D'
            url = f"https://{account_name}.blob.core.windows.net/{container_name}/{blob_name}?{sas_blob}"
            return url

blob_rul = generate_blob_sas_url('test_image.jpg')
vision_analyze_image = vision_analysis(blob_rul)
print(vision_analyze_image)


