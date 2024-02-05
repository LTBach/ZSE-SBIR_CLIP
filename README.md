## Caution
The link of above is only work for Sketchy collection. To adapt to another collection check out [Adapt](#adapt).

## Setup

1. Download [sam_ViT-B_16.pth](https://drive.google.com/file/d/1bznKsXDM5-xaUR9suCBBc7J33lIa70zJ/view?usp=sharing) at repo directory
2. Download or train the model then save the [checkpoint](https://drive.google.com/file/d/1-Ay27ghuWI3KDf7cUIV5n9RsURT6ZMGk/view?usp=sharing) to checkpoint:
```
./checkpoint
`-- best_checkpoint.pth 
```
3. Download [collection](https://drive.google.com/file/d/11GAr0jrtowTnR3otyQbNMSLPeHyvecdP/view?usp=sharing) for retrieval


4. Download [embedding of images](https://drive.google.com/file/d/1PjagAtRx2FO8l95ulCVU8WI4AdjxN2eK/view?usp=sharing) (.npz), [cluster embeddings](https://drive.google.com/file/d/1Q0RHXQ8Z0jLOHE1qmaP8CIUxCjcaKTtK/view?usp=sharing) (.npy) and [mapping cluster file](https://drive.google.com/file/d/1-Ay27ghuWI3KDf7cUIV5n9RsURT6ZMGk/view?usp=sharing) (.csv) to storage 
```
./storage
|-- all.npz
|-- centroid_embed.npy
`-- cluster.csv
```

## Running app

To use the app, running the following command:
```
streamlit run appstreamlit.py
```

## Performance and evaluate

Result compare with and without using CLIP as a global feature:

|  | mAP@100 | mPrec@100 | average_search_time |
| ---- | ---- | ---- | ---- |
| Baseline | 0.746 | 0.649 | 71.160 |
| Our approach | 0.744 | 0.634 | 1.214 |

You can check by yourself by running [evaluation script](./evaluation.py) [our result](./res/eval.csv).

## Adapt

Change path to feature then run 2 .ipynb file in [here](./adapt/)

## License
This project is released under the [MIT License](./LICENSE).

## Citation
Check out [CLIP repo](https://github.com/openai/CLIP)
Check out [ZSE-SBIR repo](https://github.com/buptLinfy/ZSE-SBIR)
