import boto3
from botocore.exceptions import ClientError
import argparse

def create_bucket(bucket_name, region=None):
    """Create an S3 bucket in a specified region

    :param bucket_name: Bucket to create
    :param region: String region to create bucket in
    :return: True if bucket created, else False
    """
    try:
        # Ensure region is not None
        if region is None:
            region = 'us-east-1'
            
        print(f"Creating bucket '{bucket_name}' in region '{region}'...")
        
        # Create the S3 client with the specified region
        s3_client = boto3.client('s3', region_name=region)
        
        # Different configuration for us-east-1 vs other regions
        if region == 'us-east-1':
            bucket_response = s3_client.create_bucket(
                Bucket=bucket_name
            )
        else:
            bucket_response = s3_client.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={
                    'LocationConstraint': region
                }
            )
        
        return True

    except ClientError as e:
        print(f"Couldn't create bucket {bucket_name}. Here's why: {e.response['Error']['Message']}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Create an S3 bucket')
    parser.add_argument('bucket_name', help='Name of the bucket to create')
    parser.add_argument('--region', 
                       default='us-east-1',
                       help='Region to create bucket in (default: us-east-1)')
    
    args = parser.parse_args()
    
    # Basic validation
    if len(args.bucket_name) < 3 or len(args.bucket_name) > 63:
        print("Error: Bucket name must be between 3 and 63 characters")
        return
    
    if not args.bucket_name.islower():
        print("Error: Bucket name must be lowercase")
        return

    # Create the bucket
    if create_bucket(args.bucket_name, args.region):
        print(f"Successfully created bucket '{args.bucket_name}' in region '{args.region}'")
    else:
        print(f"Failed to create bucket '{args.bucket_name}'")

if __name__ == "__main__":
    main()