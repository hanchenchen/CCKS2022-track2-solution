src_dir="/data"
dst_dir="/cache_ccks"

mkdir $dst_dir

cp $src_dir/pair_train.jsonl $dst_dir
cp $src_dir/pair_val.jsonl $dst_dir
cp $src_dir/item_train_info.jsonl $dst_dir
cp $src_dir/item_valid_pair.jsonl $dst_dir
cp $src_dir/item_valid_info.jsonl $dst_dir
cp $src_dir/item_test_pair.jsonl $dst_dir
cp $src_dir/item_test_info.jsonl $dst_dir

tar -xf $src_dir/item_images_384.tar -C $dst_dir