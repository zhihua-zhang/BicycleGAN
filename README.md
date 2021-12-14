## CIS680 Final Project Code Instructions

### Training
To train the bicycleGAN model, run the following command in your jupyter notebook

 `!python train.py --arg1 xxx --arg2 xxx`
 
For example,

 `!python train.py --l_kl 0.002 --l_pixel_rev 0.05 --nz 8`
 
means the weight of KL-divergence and the diversity loss are 0.002 and 0.05 separately and we assume latent dimension to be 8.

You can reproduce the final result in our report using the default setting

 `!python train.py`
 


### Inference
To generate new images using well-trained model, run 
`!python infer.py --model_pth_dir xxx --n_eval 10 --n_gen 5`

1. "model\_pth\_dir" means the directory where the models are saved. If we store our models under the "checkpoints" directory,

	* checkpoints
		* D_LR-epoch=20.pth
		* D_VAE-epoch=20.pth
		* encoder-epoch=20.pth
		* generator-epoch=20.pth

	then we should pass "--model\_pth\_dir ./checkpoints".

2. n\_eval specifies how many images we want to evaluate.
3. n\_gen refers to the number of noise-generated images.
4. Other optional arguements should be consistent with the model training settings

By default, we will evaluate 20 images and generate 8 new images.