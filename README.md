# Rockfish

Rockfish is the deep learning based tool for detecting 5mC DNA base modifications.

## Requirements

* ONT Guppy (for basecalling)
* ONT Tombo (for re-segmentation)
* Python >=3.7
* Appropriate CUDA version


## Installation

1. Clone the repository
   ```shell
   git clone ... Rockfish && cd Rockfish
   ```

2. Create virtual environment
   * Using venv

      ```shell
      python3.7 -m venv rockfish_venv
      source rockfish_venv/bin/activate
      ```

    * or conda
      ```shell
      conda create --name rockfish python=3.7
      conda activate rockfish
      ```

3. Install requirements
   ```shell
   pip install -r requirements.txt
   ```

4. Install appropriate PyTorch version (built for specific CUDA version) and pytorch-lighting

   https://pytorch.org/get-started/locally/
   
   E.g. PyTorch 1.8, pip package for CUDA 11.1

   ```shell
   pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
   pip install pytorch-lightning
   ```

## Usage

### Inference
    
    CUDA_VISIBLE_DEVICES=<cuda_devices> python src/inference.py [options ...] <model_checkpoint> <test_path> <out_path>

      <model_checkpoint>
        trained model checkpoint file
      <test_path>
        file or folder containing re-segmented fast5 files
      <out_path>
        path to the output tsv file

      options:
        -r, --recursive
          default: false
          recursively search for the input files in <test_path>  
        -b, --batch_size <int>
          default: 1024
          batch size for inference
        -t, --n_workers <int>
          default: 0
          number of workers for reading and processing data
        --prefetch_factor <int>
          default: 4
          number of batches fetched before processing, active only if n_workers > 0
        
      extraction options:
        --reseg_path <str>
          default: 'Analyses/RawGenomeCorrected_000/BaseCalled_template'
          path to the re-segmentation group in fast5
        --norm_method <str>
          default: 'standardization'
          function used for signal normalization
        --motif <str>
          default: 'CG'
          motif used to find genomic positions, regex can be used
        --sample_size <int>
          default: 20
          number of signal points for every base in k-mer
        --window <int>
          default: 8
          size of window left and right of central position, total k-mer: K = 2*W+1

### Train

1. Extract features

      ```shell
      python src/extract_features.py [options ...] <input_path> <output_path>

        <input_path> 
          file or folder containing re-segmented fast5 files
        <output_path>
          folder where the processed data will be stored

        options:
          -r, --recursive 
            default: false
            recursively search for the input files in <input_path>
          -t, --workers <int>
            default: 0
            number of workers used for data generation
          --label <int>
            default: None
            label to store for the given examples (0 or 1), not stored if not set
          --reseg_path <str>
            default: 'Analyses/RawGenomeCorrected_000/BaseCalled_template'
            path to the re-segmentation group in fast5
          --norm_method <str>
            default: 'standardization'
            function used for signal normalization
          --motif <str>
            default: 'CG'
            motif used to find genomic positions, regex can be used
          --sample_size <int>
            default: 20
            number of signal points for every base in k-mer
          --window <int>
            default: 8
            size of window left and right of central position, total k-mer: K = 2*W+1
          --bed_path <str>
            default: None
            path to the bedmethyl file used for position filtering
          --bed_filter <str>
            default: None
            filter method if bed_path is set, currently implemented only high_confidence (n_reads >= 10, meth_freq (label) either 0 (0) or 100 (1))
      ```

2. Train Rockfish

   ```shell
   CUDA_VISIBLE_DEVICES=<cuda_devices> python src/train.py [options ...] <train_path> <val_path>

     <train_path>
       file containing extracted data in binary format used for training
     <val_path>
       file containing extracted data in binary format used for validation
     
     options:
       -b, --train_batch_size <int>
         default: 128
         mini-batch size used for training (effective batch size is n_gpu * train_batch_size)
       --val_batch_size <int>
         default: 1024
         mini-batch size used for validation (effective n_gpu * val_batch_size)
       --iterable
         default: false
         instead of loading data in RAM, iteratively fetch data (not recommended)

     model options:
       --dropout <float>
         default: 0.1
         dropout value used for training
       --nhead <int>
         default: 8
         number of heads in multi-head attention
       --dim_ff <int>
         default: 1024
         dimension of hidden layer in feed-forward network (in transformer)
       --nlayers <int>
         default: 6
         number of transformer encoder layers

     train options:
       --epochs <int>
         default: 30
         number of training epochs
       --wd <float>
         default: 1e-4
         weight decay used in AdamW
       --lr <float>
         default: 1e-4
         learning rate upper bound for CyclicLR
       --step_size_up <int>
         default: None
         number of iterations for half cycle in CyclicLR, inferred from train dataset size if not set 
   ```

## Acknowledgment

This work has been supported in part by AI Singapore under the project the Deep Generative Modeling of Epigenomics Data (AISG-RPKS-2019-001), by Croatian Science Foundation under the project Single genome and metagenome assembly (IP-2018-01-5886), by the A\*STAR Computational Resource Centre and by the National Supercomputing Centre, Singapore through the use of their high-performance computing facilities. D.S. and M.Š. have been partially supported by funding from A\*STAR, Singapore.