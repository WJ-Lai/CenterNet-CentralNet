
# ini
conda activate CenterNet-new && cd ~/Code/CenterNet-cuda10-multi-spectral/src/

rm -rf ~/Code/CenterNet-cuda10-multi-spectral/data/rgb/ && rm -rf ~/Code/CenterNet-cuda10-multi-spectral/data/fir/ && rm -rf ~/Code/CenterNet-cuda10-multi-spectral/data/mir/ && rm -rf ~/Code/CenterNet-cuda10-multi-spectral/data/nir/ && cp -r ~/Data/ir_det_dataset/ir_det_dataset_COCO/* ~/Code/CenterNet-cuda10-multi-spectral/data/

cd /home/vincent/Code/CenterNet-cuda10-multi-spectral/src/


# train
batch_size=12 && num_epochs=10 && sensor="rgb"

batch_size=12
num_epochs=300
sensor="rgb"

sensor="rgb" && python main.py ctdet --exp_id coco_dla --batch_size $batch_size --lr 1.25e-4 --gpus 0 --num_workers 0 --num_epochs $num_epochs --dataset $sensor && sensor="fir" && python main.py ctdet --exp_id coco_dla --batch_size $batch_size --lr 1.25e-4 --gpus 0 --num_workers 0 --num_epochs $num_epochs --dataset $sensor && sensor="mir" && python main.py ctdet --exp_id coco_dla --batch_size $batch_size --lr 1.25e-4 --gpus 0 --num_workers 0 --num_epochs $num_epochs --dataset $sensor &&
sensor="nir" && python main.py ctdet --exp_id coco_dla --batch_size $batch_size --lr 1.25e-4 --gpus 0 --num_workers 0 --num_epochs $num_epochs --dataset $sensor 

sensor="rgb" && python main.py ctdet --exp_id coco_dla --batch_size $batch_size --lr 1.25e-4 --gpus 0 --num_workers 0 --num_epochs $num_epochs --dataset $sensor

# continue train
batch_size=12
num_epochs=300
sensor="rgb"
python main.py ctdet --exp_id coco_dla --batch_size $batch_size --lr 1.25e-4 --gpus 0 --num_workers 0 --num_epochs $num_epochs --dataset $sensor --load_model /home/vincent/Code/CenterNet-cuda10-multi-spectral/exp/ctdet/coco_dla/$sensor/model_last.pth

# test
sensor="rgb" && python test.py --exp_id coco_dla --not_prefetch_test ctdet --dataset $sensor --load_model /home/vincent/Code/CenterNet-cuda10-multi-spectral/exp/ctdet/coco_dla/$sensor/model_best.pth

sensor="rgb" && python test.py --exp_id coco_dla --not_prefetch_test ctdet --dataset $sensor --load_model /home/vincent/Code/CenterNet-cuda10-multi-spectral/exp/ctdet/coco_dla/$sensor/model_best.pth && sensor="fir" && python test.py --exp_id coco_dla --not_prefetch_test ctdet --dataset $sensor --load_model /home/vincent/Code/CenterNet-cuda10-multi-spectral/exp/ctdet/coco_dla/$sensor/model_best.pth && sensor="mir" && python test.py --exp_id coco_dla --not_prefetch_test ctdet --dataset $sensor --load_model /home/vincent/Code/CenterNet-cuda10-multi-spectral/exp/ctdet/coco_dla/$sensor/model_best.pth && sensor="nir" && python test.py --exp_id coco_dla --not_prefetch_test ctdet --dataset $sensor --load_model /home/vincent/Code/CenterNet-cuda10-multi-spectral/exp/ctdet/coco_dla/$sensor/model_best.pth


sensor="rgb" && python test.py --exp_id coco_dla --not_prefetch_test ctdet --dataset $sensor --load_model /home/vincent/Code/CenterNet-cuda10-multi-spectral/exp/ctdet/coco_dla/$sensor/model_last.pth && sensor="fir" && python test.py --exp_id coco_dla --not_prefetch_test ctdet --dataset $sensor --load_model /home/vincent/Code/CenterNet-cuda10-multi-spectral/exp/ctdet/coco_dla/$sensor/model_last.pth && sensor="mir" && python test.py --exp_id coco_dla --not_prefetch_test ctdet --dataset $sensor --load_model /home/vincent/Code/CenterNet-cuda10-multi-spectral/exp/ctdet/coco_dla/$sensor/model_last.pth && sensor="nir" && python test.py --exp_id coco_dla --not_prefetch_test ctdet --dataset $sensor --load_model /home/vincent/Code/CenterNet-cuda10-multi-spectral/exp/ctdet/coco_dla/$sensor/model_last.pth


# show

sensor="rgb" && python demo.py ctdet --demo /home/vincent/Code/CenterNet-cuda10-multi-spectral/data/$sensor/images --dataset $sensor --load_model /home/vincent/Code/CenterNet-cuda10-multi-spectral/exp/ctdet/coco_dla/$sensor/model_last.pth

sensor="rgb" && python demo.py ctdet --demo /home/vincent/Code/CenterNet-cuda10-multi-spectral/data/$sensor/images --dataset $sensor --load_model /home/vincent/Code/CenterNet-cuda10-multi-spectral/exp/ctdet/coco_dla/$sensor/model_last.pth && sensor="fir" && python demo.py ctdet --demo /home/vincent/Code/CenterNet-cuda10-multi-spectral/data/$sensor/images --dataset $sensor --load_model /home/vincent/Code/CenterNet-cuda10-multi-spectral/exp/ctdet/coco_dla/$sensor/model_last.pth && sensor="mir" && python demo.py ctdet --demo /home/vincent/Code/CenterNet-cuda10-multi-spectral/data/$sensor/images --dataset $sensor --load_model /home/vincent/Code/CenterNet-cuda10-multi-spectral/exp/ctdet/coco_dla/$sensor/model_last.pth && sensor="nir" && python demo.py ctdet --demo /home/vincent/Code/CenterNet-cuda10-multi-spectral/data/$sensor/images --dataset $sensor --load_model /home/vincent/Code/CenterNet-cuda10-multi-spectral/exp/ctdet/coco_dla/$sensor/model_last.pth


sensor="rgb" && python demo.py ctdet --demo /home/vincent/Code/CenterNet-cuda10-multi-spectral/data/$sensor/images --dataset $sensor --load_model /home/vincent/Code/CenterNet-cuda10-multi-spectral/exp/ctdet/coco_dla/$sensor/model_best.pth && sensor="fir" && python demo.py ctdet --demo /home/vincent/Code/CenterNet-cuda10-multi-spectral/data/$sensor/images --dataset $sensor --load_model /home/vincent/Code/CenterNet-cuda10-multi-spectral/exp/ctdet/coco_dla/$sensor/model_best.pth && sensor="mir" && python demo.py ctdet --demo /home/vincent/Code/CenterNet-cuda10-multi-spectral/data/$sensor/images --dataset $sensor --load_model /home/vincent/Code/CenterNet-cuda10-multi-spectral/exp/ctdet/coco_dla/$sensor/model_best.pth && sensor="nir" && python demo.py ctdet --demo /home/vincent/Code/CenterNet-cuda10-multi-spectral/data/$sensor/images --dataset $sensor --load_model /home/vincent/Code/CenterNet-cuda10-multi-spectral/exp/ctdet/coco_dla/$sensor/model_best.pth










sensor="fir" && python main.py ctdet --exp_id coco_dla --batch_size $batch_size --lr 1.25e-4 --gpus 0 --num_workers 0 --num_epochs $num_epochs --dataset $sensor && sensor="mir" && python main.py ctdet --exp_id coco_dla --batch_size $batch_size --lr 1.25e-4 --gpus 0 --num_workers 0 --num_epochs $num_epochs --dataset $sensor && sensor="nir" && python main.py ctdet --exp_id coco_dla --batch_size $batch_size --lr 1.25e-4 --gpus 0 --num_workers 0 --num_epochs $num_epochs --dataset $sensor && batch_size=12 && num_epochs=50 && sensor="rgb" && python main.py ctdet --exp_id coco_dla --batch_size $batch_size --lr 1.25e-4 --gpus 0 --num_workers 0 --num_epochs $num_epochs --dataset $sensor

batch_size=12 && num_epochs=30 && sensor="nir" && python main.py ctdet --exp_id coco_dla --batch_size $batch_size --lr 1.25e-4 --gpus 0 --num_workers 0 --num_epochs $num_epochs --dataset $sensor --load_model /home/vincent/Code/CenterNet-cuda10-multi-spectral/exp/ctdet/coco_dla/$sensor/model_last.pth && batch_size=12 && num_epochs=300 && sensor="rgb" && python main.py ctdet --exp_id coco_dla --batch_size $batch_size --lr 1.25e-4 --gpus 0 --num_workers 0 --num_epochs $num_epochs --dataset $sensor

每一句都要加--dataset


sensor="mir" && python demo.py ctdet --demo /home/vincent/Code/CenterNet-cuda10-multi-spectral/data/$sensor/images --dataset $sensor --load_model /home/vincent/Code/CenterNet-cuda10-multi-spectral/exp/ctdet/coco_dla/$sensor/model_best.pth --dataset $sensor 

&& 

sensor="nir" && python demo.py ctdet --demo /home/vincent/Code/CenterNet-cuda10-multi-spectral/data/$sensor/images --dataset $sensor --load_model /home/vincent/Code/CenterNet-cuda10-multi-spectral/exp/ctdet/coco_dla/$sensor/model_best.pth --dataset $sensor



sensor="rgb" && python demo.py ctdet --demo /home/vincent/Code/CenterNet-cuda10-multi-spectral/data/$sensor/images --dataset $sensor --load_model /home/vincent/Code/CenterNet-cuda10-multi-spectral/exp/ctdet/coco_dla/$sensor/model_best.pth --dataset $sensor



--exp_id 实验名称
--fuse 选择不同的传感器