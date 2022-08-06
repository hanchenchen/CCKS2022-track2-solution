src_dir="/data"
dst_dir="/cache_ccks"

mkdir $dst_dir
mount -t tmpfs -o size=50G tmpfs $dst_dir

unzip -q $src_dir/item_train_images.zip -d $dst_dir
unzip -q $src_dir/item_valid_images.zip -d $dst_dir
unzip -q $src_dir/item_test_images.zip -d $dst_dir

python src/data/preprocess/resize_img.py --src_dir $dst_dir --dst_dir $dst_dir/item_images --img_size 384

(cd $dst_dir && tar -cf item_images_384.tar item_images)
mv $dst_dir/item_images_384.tar $src_dir