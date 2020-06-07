branch=`git rev-parse --abbrev-ref HEAD`
project=`gcloud config get-value project 2> /dev/null`

# Copy shared resources in
cp common/* $1
cp -r ../src/knn $1

# Submit build from within subdirectory
gcloud config set builds/use_kaniko True
(cd $1; gcloud builds submit --tag gcr.io/$project/mihir-$1-$branch)

# Remove shared resources
for file in $(ls common/)
do
   rm $1/"$file"
done
rm -rf $1/knn

# Deploy Cloud Run handler
gcloud run deploy mihir-$1-$branch --image gcr.io/$project/mihir-$1-$branch --platform managed --concurrency 1 --cpu 2 --max-instances 1000 --memory 2Gi --timeout 900 --region us-central1 --allow-unauthenticated
