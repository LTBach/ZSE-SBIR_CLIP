## Setup

1. Download or train the model then save the checkpoint to checkpoint:
```
./checkpoint
`-- best_checkpoint.pth
```
2. Download embedding of image (.npz), cluster embeddings (.npy) and mapping cluster file (.csv) to storage 
```
./storage
|-- all.npz
|-- centroid_embed.npy
`-- cluster.csv
```

## Running

To use the app, running the following command:
```
streamlit run appstreamlit.py
```


## License
This project is released under the [MIT License](./LICENSE).
