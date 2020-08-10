## Data preparation
We adapt the data preparation codes from [Graph-CMR](https://github.com/nkolot/GraphCMR/tree/master/datasets/preprocess).To use this functionality, you need to download the relevant datasets.
The datasets that our code supports are:
1. [Human3.6M](http://vision.imar.ro/human3.6m/description.php)
2. [UP-3D](http://files.is.tuebingen.mpg.de/classner/up/)
3. [LSP](http://sam.johnson.io/research/lsp.html)
4. [MPII](http://human-pose.mpi-inf.mpg.de)
5. [COCO](http://cocodataset.org/#home)

More specifically:
1. **Human3.6M**: Unfortunately, due to license limitations, we are not allowed to redistribute the MoShed data that we used for training. We only provide code to evaluate our approach on this benchmark. To download the relevant data, please visit the [website of the dataset](http://vision.imar.ro/human3.6m/description.php) and download the Videos, BBoxes MAT (under Segments) and 3D Positions Mono (under Poses) for Subjects S9 and S11. After downloading and uncompress the data, store them in the folder ```${Human3.6M root}```. The sructure of the data should look like this:
```
${Human3.6M root}
|-- S9
    |-- Videos
    |-- Segments
    |-- Bboxes
|-- S11
    |-- Videos
    |-- Segments
    |-- Bboxes
```
You also need to edit the file ```utils.config.py``` to reflect the path ```${Human3.6M root}``` you used to store the data.

2. UP-3D: We use this data both for training and evaluation. You need to download the [UP-3D zip](http://files.is.tuebingen.mpg.de/classner/up/datasets/up-3d.zip) (that provides images and 3D shapes for training and testing) and the [UPi-S1h zip](http://files.is.tuebingen.mpg.de/classner/up/datasets/upi-s1h.zip) (which we will need for silhouette evaluation on the LSP dataset). After you unzip, please edit ```config.py``` to include the paths for the two datasets.

3. LSP: We again use LSP both for training and evaluation. You need to download the high resolution version of the dataset [LSP dataset original](http://sam.johnson.io/research/lsp_dataset_original.zip) (for training) and the low resolution version [LSP dataset](http://sam.johnson.io/research/lsp_dataset.zip) (for evaluation). After you unzip the dataset files, please complete the relevant root paths of the datasets in the file ```config.py```.

4. MPII: We use this dataset for training. You need to download the compressed file of the [MPII dataset](https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz). After uncompressing, please complete the root path of the dataset in the file ```config.py```.

5. COCO: We use this dataset for training. You need to download the [images](http://images.cocodataset.org/zips/train2014.zip)
 and the [annotations](http://images.cocodataset.org/annotations/annotations_trainval2014.zip) 
 for the 2014 training set of the dataset. After you unzip the files, the folder structure should look like:
```
${COCO root}
|-- train2014
|-- annotations
```
Then, you need to edit the ```utils.config.py``` file with the ```${COCO root}``` path.

### Generate dataset files
After preparing the data, we continue with the preprocessing to produce the data/annotations for each dataset in the expected format. You need to run the file ```preprocess_datasets.py``` from the main folder of this repo that will do all this work automatically. Depending on whether you want to do evaluation or/and training, we provide two modes:

If you want to generate the files such that you can evaluate our pretrained models, you need to run:
```
python preprocess_datasets.py --eval_files
```

If you want to generate the files such that you can train using the supported datasets, you need to run:
```
python preprocess_datasets.py --train_files
```

### Generate ground truth IUV image
For the training process, we also need to generate the GT IUV image using the GT SMPL parameters.
You need to run:
```
python preprocess_datasets.py --gt_iuv
```
Above command will generate the IUV image under our new UV map. 
If you want to generate the IUV image under SMPL default UV map, you may run:
```
python preprocess_datasets.py --gt_iuv --uv_type=SMPL
```
### Extra datasets preparation
We also provide the code to train and evaluate our model on some extra datasets: 
1. [SURREAL](https://www.di.ens.fr/willow/research/surreal/data/)
2. [3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/)
3. [MPI-INF-3DHP](http://gvv.mpi-inf.mpg.de/3dhp-dataset/)
4. [HR-LSPET](http://sam.johnson.io/research/lspet.html)

**Download Data**
1. **SURREAL**: We use SURREAL dataset for train and evaluation.
You need to download the data from [dataset website](https://www.di.ens.fr/willow/research/surreal/data/),
and then complete the root path of the dataset in the file utils.config.py. 
For the evaluation on SURREAL dataset, we use the same setting as [BodyNet](https://www.di.ens.fr/willow/research/bodynet/).
In BodyNet, not all eval image is used, so you may need to download the 
valid image list from [here](https://drive.google.com/drive/folders/1xWBVfQa7OZ14VgT9BVO9Lj_kDqRAcQ-e).
The valid image list is gotten from [this issue](https://github.com/gulvarol/bodynet/issues/5).

2. **3DPW**: We use this dataset only for evaluation. You need to download the data from the 
[dataset website](https://virtualhumans.mpi-inf.mpg.de/3DPW/). 
After you unzip the dataset files, please complete the root path of the dataset in the file ```utils.config.py```.

3. **MPI-INF-3DHP**: We use this dataset for training and evaluation.
 You need to download the data from the [dataset website](http://gvv.mpi-inf.mpg.de/3dhp-dataset). 
 The expected fodler structure at the end of the processing looks like:
```
${MPI_INF_3DHP root}
|-- mpi_inf_3dhp_test_set
    |-- TS1
|-- S1
    |-- Seq1
        |-- imageFrames
            |-- video_0
```
Then, you need to edit the ```utils.config.py``` file with the ```${MPI_INF_3DHP root}``` path.

4. **HR-LSPET**: We use the extended training set of LSP in its high resolution form (HR-LSPET).
 You need to download the [high resolution images](http://datasets.d2.mpi-inf.mpg.de/hr-lspet/hr-lspet.zip). 
 After you unzip the dataset files, please complete the root path of the dataset in the file ```utils.config.py```.
 
**Generate Dataset Files**

In order to generate the dataset files, you may run:
```
python preprocess_extra_datasets.py --eval_files --train_files
```
For 3DPW, MPI-INF-3DHP and 3DPW dataset, you may also 
directly use the processed dataset files provided by SPIN. You may run;
```
wget http://visiondata.cis.upenn.edu/spin/dataset_extras.tar.gz
```
And then unzip the files to the directory containing your dataset files.

**Use SPIN Fitting Results as GT**

If you want to use the fitting results of SPIN as GT, you can download 
the fitting results from 
[here](http://visiondata.cis.upenn.edu/spin/spin_fits.tar.gz) and 
unzip the fits to the directory containing your dataset files.


**Generate GT IUV Images**

 You may generate the GT IUV images by run:
```
python preprocess_extra_datasets.py --gt_iuv
```