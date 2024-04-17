#!/bin/bash

set -e

for i in {0..7}
do
  python run.py experiment=classification_embeddings_ICPR name=embedding_classification_mlp-hidden-2048-3-small +datamodule.embedding=aws_text-embedding-3-small logger.wandb.tags=[MLP,embedding,3-small]
  python run.py experiment=classification_embeddings_ICPR name=embedding_classification_mlp-hidden-2048-v2-base +datamodule.embedding=aws_jina-embeddings-v2-base-en logger.wandb.tags=[MLP,embedding,v2-base]
  python run.py experiment=classification_embeddings_ICPR name=embedding_classification_mlp-hidden-2048-mistral +datamodule.embedding=aws_mistral-embed logger.wandb.tags=[MLP,embedding,mistral]
  python run.py experiment=classification_embeddings_ICPR name=embedding_classification_mlp-hidden-2048-3-large +datamodule.embedding=aws_text-embedding-3-large logger.wandb.tags=[MLP,embedding,3-large]
  python run.py experiment=classification_embeddings_ICPR name=embedding_classification_mlp-hidden-2048-ada-002 +datamodule.embedding=ws_text-embedding-ada-002 logger.wandb.tags=[MLP,embedding,ada-002]
done