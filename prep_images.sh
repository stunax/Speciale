

rm -rf ~/data/prep_images/*
rm ~/preped_images.zip

python3 ~/Speciale/labels/prepare_new_images.py
cd ~/data
zip -r ~/preped_images.zip prep_images/
cd -
