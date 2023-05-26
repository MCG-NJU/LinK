# Training

We provide two shell files, ```start_multigpu.sh``` and ```train.sh```,  to start the training process. One can start the training with 

```bash
sh train.sh
```


In ```train.sh```, 
> [*-t*] or [*--target*] determines the experiment name. \
> [*--gpus*] controls the number of gpus.

In ```start_multigpu.sh```,
> [*dataset*] determines using which dataset: SemanticKITTI or nuScenes. \
> [*model*] means to use which backbone: linkunet, linkencoder, minkunet, spvcnn. 

> ```linkencoder``` is a decoder-free architecture, more efficient.

We backup the full runtime code into `runs/`. One can recover the runtime code in `runs/${dataset}/${model}/${experiment name}/`.


# Evaluation

## Reproduction

For a quick rerpduction, we provide the full run-time code along with the checkpoint. 


After finishing the installation and dataset preparation, download the log file in model zoo, and uncompress them into the ```segmentation/runs``` folder. Take the ```cos_x[2x3]``` as an example. Make a hyperlink to link the data folder to the ```backup``` path. 

```bash
cd runs/cos_x[2x3_g1]/2023-05-23_15:30:26/backup/
ln -s ../../../../data .
```

The file structure should be like:
```bash
LinK/segmentation/runs/
        cos_x[2x3_g1]/
            2023-05-23_15:30:26/
                ious.txt
                backup/
                    data/
                        SemanticKITTI/
                        nuScenes/
                    evaluate.py
                    evaluate.sh # validating on sequence 08
                    test.py
                    test.sh # generating results on test split
                    ...
                checkpoints/
                    max-iou-val.pt
                logging/
                tensorboard/
                summary/
                metainfo/
```

Then, run the evaluation shell:

```bash
chomod +x evaluation.sh
./evaluation.sh
```



