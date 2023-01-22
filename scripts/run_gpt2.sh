output_dir="output/"
model_name="EleutherAI/gpt-j-6B"

echo "${model_name} for English track"
python src/run_gpt2.py --task_dir data/tasks --data_dir data/splits/default --max_num_instances_per_task 1 --max_num_instances_per_eval_task 10  --num_pos_examples 2 --num_neg_examples 0  --output_dir ${output_dir}/default/${model_name} --model_name ${model_name}
python src/compute_metrics.py --predictions ${output_dir}/default/${model_name}/predicted_examples.jsonl --track default
