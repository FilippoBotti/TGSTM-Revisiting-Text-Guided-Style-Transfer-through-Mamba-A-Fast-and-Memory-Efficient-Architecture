# Revisiting Text-Guided Style Transfer through Mamba: A Fast and Memory-Efficient Architecture
*Authors: Filippo Botti, Alex Ergasti, Tomaso Fontanini and Andrea Prati*
<br>[ArXiv](...)

This repository is the official implementation of Revisiting Text-Guided Style Transfer through Mamba: A Fast and Memory-Efficient Architecture
This paper explores a novel design of Mamba to perform text-guided style transfer.

## Results presentation 

<p align="center">
<img src="https://github.com/FilippoBotti/TGSTM-Revisiting-Text-Guided-Style-Transfer-through-Mamba-A-Fast-and-Memory-Efficient-Architecture/blob/main/resources/results.png" width="90%" height="90%">
</p>
Examples of generated images from our Mamba model given a caption and a content image. <br>


## Framework
<p align="center">
<img src="https://github.com/FilippoBotti/Revisiting-Text-Guided-Style-Transfer-through-Mamba-A-Fast-and-Memory-Efficient-Architecture/resources/arch.png" width="100%" height="100%">
</p> 
a) Full architecture. It takes as input a content image and a style caption and generates the content image stylized as the style caption. b)
Our Mamba Decoder, which takes both style and content as input. In particular it generates matrices ∆ and B from style embedding, while C from
content features and merge them together in the Selective Scan laye

## Experiment
### Requirements
In order to run the project please install the environment by following these commands: 
```
conda create -n TGSTM
pip install -r req.txt
conda activate TGSTM
```

You can find the train and test images used inside ./data folder.
Please modify all the .sh files with the correct path for your checkpoints and images before 
running the following instructions.

### Testing
```
sh scripts/test.sh
```

### Training  
```
sh scripts/train.sh
```

## Code explanation
The full model (fig. 2(a)) can be found at [StyTr.py](https://github.com/FilippoBotti/Revisiting-Text-Guided-Style-Transfer-through-Mamba-A-Fast-and-Memory-Efficient-Architecture/models/StyTr.py). In this file you can find the whole architecture. <br>
The Mamba Decoder (fig. 2 (b) module can be found at [mamba.py](https://github.com/FilippoBotti/Revisiting-Text-Guided-Style-Transfer-through-Mamba-A-Fast-and-Memory-Efficient-Architecture/models/mamba.py) <br>
Finally, our VSSM's implementation (both with a single input and with two input merged for style transfer) can be found at [single_direction_vssm.py](https://github.com/FilippoBotti/Revisiting-Text-Guided-Style-Transfer-through-Mamba-A-Fast-and-Memory-Efficient-Architecture/vssm/mamba_arch.py).

<!-- ### Reference -->
<!-- If you find our work useful in your research, please cite our paper using the following BibTeX entry ~ Thank you ^ . ^. Paper Link [pdf](https://www.arxiv.org/abs/2409.10385)<br>  -->


<!-- ```
@inproceedings{botti2025mamba,
  author={Botti, Filippo and Ergasti, Alex and Rossi, Leonardo and Fontanini, Tomaso and Ferrari, Claudio and Bertozzi, Massimo and Prati, Andrea},
  booktitle={2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)}, 
  title={Mamba-ST: State Space Model for Efficient Style Transfer}, 
  year={2025},
  volume={},
  number={},
  pages={7797-7806},
  keywords={Measurement;Fuses;Computational modeling;Memory management;Transformers;Diffusion models;Mathematical models;State-space methods;Time complexity;Streams;mamba;style transfer;state space model},
  doi={10.1109/WACV61041.2025.00757}
}
``` -->

### Acknowledgments
Our code is inspired by [StyTR-2](https://github.com/diyiiyiii/StyTR-2) and [MambaST](https://github.com/FilippoBotti/MambaST).
