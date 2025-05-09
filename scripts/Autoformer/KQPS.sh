export CUDA_VISIBLE_DEVICES=0

model_name=Autoformer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Kuai/ \
  --data_path easy_train_data.csv \
  --model_id Easy_QPS_1440_60 \
  --model $model_name \
  --data KuaiEasyQPS \
  --features S \
  --seq_len 1440 \
  --label_len 48 \
  --pred_len 60 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --freq 'b' \
  --des 'Exp' \
  --itr 1