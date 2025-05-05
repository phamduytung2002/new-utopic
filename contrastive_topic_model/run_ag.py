import os

bsz = 200
dataset = 'agnews'
# n_topic = 50
epochs_1 = 1000
epochs_2 = 100
bert_model = 'sentence-transformers/all-MiniLM-L6-v2'

if __name__ == "__main__":
    for n_topic in [20, 70]:
        # for stage_1_dist in [1e6, 1e5, 1e4, 1e3, 1e2, 1e1, 1]:
            # print("*******************************")
            # print(f'Running {dataset} with stage_1_dist={stage_1_dist}')
            # os.system(f'python runs1.py --base-model {bert_model} --dataset {dataset} --n-word 5000 --epochs-1 {epochs_1} --bsz {bsz} --coeff-1-dist {stage_1_dist} --n-cluster {n_topic} --gpus {os.environ["CUDA_VISIBLE_DEVICES"]}')

        for stage_1_dist in [1e6, 1e5, 1e4, 1e3, 1e2, 1e1, 1]:
            for stage_2_cons in [1]:
                for stage_2_dist in [0.1, 1, 10]:
                    print("*******************")
                    print("Running stage 2 with:")
                    print(f"stage_2_cons={stage_2_cons}")
                    print(f"stage_2_dist={stage_2_dist}")
                    os.system(f'python runs23.py --base-model {bert_model} --dataset {dataset} --n-word 5000 --epochs-1 {epochs_1} --epochs-2 {epochs_2} --bsz {bsz} --stage-2-lr 2e-2 --stage-2-repeat 3 --coeff-1-dist {stage_1_dist} --coeff-2-cons {stage_2_cons} --coeff-2-dist {stage_2_dist} --n-cluster {n_topic}')
