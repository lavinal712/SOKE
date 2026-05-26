# python -m scripts.get_motion_code --cfg configs/soke_local.yaml --nodebug --use_gpus 0 --device 0
python -m train --cfg configs/soke.yaml --nodebug --use_gpus 0,1,2,3,4,5
python -m train --cfg configs/signspark.yaml --nodebug --use_gpus 0,1,2,3,4,5,6,7
