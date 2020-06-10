branch=`git rev-parse --abbrev-ref HEAD`
project=`gcloud config get-value project 2> /dev/null`

# Deploy Cloud Run handler
gcloud run deploy mihir-$1-$branch --image gcr.io/$project/mihir-$1-$branch --platform managed --concurrency 1 --cpu 1 --max-instances 1000 --memory 2Gi --timeout 900 --region us-central1 --allow-unauthenticated
