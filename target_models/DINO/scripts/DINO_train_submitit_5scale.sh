coco_path=$1
python run_with_submitit.py --timeout 3000 --job_name DINO \
	--job_dir logs/DINO/R50-MS5-%j --ngpus 8 --nodes 2 \
	-c config/DINO/DINO_5scale.py --coco_path $coco_path \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0
