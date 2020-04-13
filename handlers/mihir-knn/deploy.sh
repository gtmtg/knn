gcloud builds submit --tag gcr.io/mihir-knn/knn
gcloud run deploy knn --image gcr.io/mihir-knn/knn --platform managed --concurrency 1 --cpu 1 --max-instances 1000 --memory 2Gi --timeout 900 --region us-west1
