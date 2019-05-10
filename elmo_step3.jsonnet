local embedding_dim = 256;
local hidden_dim = 128;
local batch_size = 60;
local num_epochs = 10;
local patience = 5;
local encoder_hidden_dim = 8;
local num_layers = 2;
local projection_dim = 8;
local feedforward_hidden_dim = 8;
local num_attention_heads = 2;

{
  "dataset_reader": {
    "type": "sst_tokens",
    "token_indexers": {
      "tokens": {
        "type": "elmo_characters"
      }
    }
  },
  "train_data_path": "trees/train.txt",
  "validation_data_path": "trees/dev.txt",

  "model": {
    "type": "lstm_classifier",

    "word_embeddings": {
      "tokens": {
        "type": "elmo_token_embedder",
        "options_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json",
        "weight_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5",
        "do_layer_norm": false,
        "dropout": 0.5
      }
    },

    "encoder": {
      "type": "transformer-encoder",
      "input_dim": embedding_dim,
      "hidden_dim": encoder_hidden_dim,
      "num_layers": num_layers,
      "projection_dim": projection_dim,
      "feedforward_hidden_dim": feedforward_hidden_dim,
      "num_attention_heads": num_attention_heads
    }
  },
  "iterator": {
    "type": "bucket",
    "batch_size": batch_size,
    "sorting_keys": [["tokens", "num_tokens"]]
  },
  "trainer": {
    "optimizer": "adam",
    "num_epochs": num_epochs,
    "patience": patience
  }
}

