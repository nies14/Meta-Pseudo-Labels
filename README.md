# Meta-Pseudo-Labels


For running it in GCP

-open gcp console and type the bellow

export PROJECT_ID=project-id
gcloud config set project $PROJECT_ID
gcloud config set project $PROJECT_ID
gsutil mb -p ${PROJECT_ID} -c standard -l us-central1 -b on gs://bucket-name

ctpu up --project=${PROJECT_ID} \
 --zone=us-central1-b \
 --tf-version=2.3.1 \
 --name=tpu-quickstart
 
gcloud compute ssh tpu-quickstart --zone=us-central1-b
 
export STORAGE_BUCKET=gs://bucket-name
 
export TPU_NAME=tpu-quickstart
export MODEL_DIR=$STORAGE_BUCKET/mnist
DATA_DIR=$STORAGE_BUCKET/data
export PYTHONPATH="$PYTHONPATH:/usr/share/models"
export output_dir="output"
pip install -r requirements.txt
git clone https://github.com/nies14/Meta-Pseudo-Labels.git

python3 meta_pseudo_labels/ -m main.py \
  --task_mode="train" \
  --dataset_name="cifar10_4000_mpl" \
  --output_dir=$output_dir \
  --model_type='resnet-50' \
  --log_every=100 \
  --image_size=32 \
  --num_classes=10 \
  --optim_type="momentum" \
  --lr_decay_type="cosine" \
  --use_bfloat16 \
  --use_tpu \
  --nouse_augment \
  --reset_output_dir \
  --eval_batch_size=64 \
  --alsologtostderr \
  --running_local_dev \
  --train_batch_size=128 \
  --label_smoothing=0.1 \
  --grad_bound=5. \
  --uda_data=15 \
  --uda_steps=50000 \
  --uda_temp=0.5 \
  --uda_threshold=0.6 \
  --uda_weight=20. \
  --tpu_platform=tpu_platform \
  --tpu_topology=tpu_topology \
  --train_batch_size=1024 \
  --num_train_steps=700000 \
  --num_warmup_steps=10000 \
  --use_augment=False \
  --augment_magnitude=17 \
  --batch_norm_batch_size=1024 \
  --mpl_student_lr=0.1 \
  --mpl_student_lr_wait_steps=20000 \
  --mpl_student_lr_warmup_steps=5000 \
  --mpl_teacher_lr=0.15 \
  --mpl_teacher_lr_warmup_steps=5000 \
  --ema_decay=0.999 \
  --dense_dropout_rate=0.1 \
  --weight_decay=1e-4 \
  --save_every=1000
  
if u get tensorflow.contrib error then type 'sudo -H python3.7 -m pip install tensorflow==1.15.3'  
