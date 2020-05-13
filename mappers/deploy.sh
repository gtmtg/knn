# Copy shared resources in
cp common/* $1
cp -r ../src/knn $1

# Submit build from within subdirectory
gcloud config set builds/use_kaniko True
(cd $1; gcloud builds submit --tag gcr.io/visualdb-1046/mihir-$1)

# Remove shared resources
for file in $(ls common/)
do
   rm $1/"$file"
done
rm -rf $1/knn

# Deploy Cloud Run handler
gcloud run deploy mihir-$1 --image gcr.io/visualdb-1046/mihir-$1 --platform managed --concurrency 1 --cpu 1 --max-instances 1000 --memory 2Gi --timeout 900 --region us-west1 --allow-unauthenticated
