%%file train.sh
python -m sparsify openai-community/gpt2 NeelNanda/pile-10k --ctx_len 128 --filter_bos=True \
--transcode=True --skip_connection=True \
--matryoshka=True \
--matryoshka_expansion_factors 8 16 32 \
--k=32 \
--batch_size=4 \
--activation=batchtopk \
--hookpoints h.0.mlp h.1.mlp h.2.mlp h.3.mlp h.4.mlp h.5.mlp h.6.mlp h.7.mlp h.8.mlp h.9.mlp h.10.mlp h.11.mlp \
--run_name gpt2-matroshka-sweep \
--cross_layer=12 --coalesce_topk=concat --topk_coalesced=False \
--post_encoder_scale=True --normalize_io=True \
--lr 3e-4 \
--optimizer adam --lr_warmup_steps 50 \
--auxk_alpha 0.1 \
--dead_feature_threshold 1000000 