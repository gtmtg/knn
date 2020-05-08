# Copy shared resources in
cp common/* $1
cp ../src/knn $1

# (cd $1; gcloud builds submit --tag gcr.io/visualdb-1046/mihir-$1)

# Remove shared resources
for file in common/
do
   rm $1/"$file"
done
rm -rf $1/src

# gcloud run deploy $1 --image gcr.io/visualdb-1046/$1 --platform managed --concurrency 1 --cpu 1 --max-instances 1000 --memory 2Gi --timeout 900 --region us-central1 --allow-unauthenticated
